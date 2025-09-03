#!/usr/bin/env python3
import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils.parse_class import literal_eval_with_self, extract_class_attr_paths


class Keys:
    """Symbol names used during extraction."""
    SAMPLES = "samples"
    DS_LIST = "ds_list"
    TGT_LIST = "tgt_list"
    OP = "op"
    TEXT = "text"
    TARGET = "target"
    TEST_PREFIX = "test"
    SELF = "self"
    SETUP = "setUp"


@dataclass
class MethodInfo:
    """Normalized info for a test method."""
    op_code: Optional[str] = None
    ds: Optional[str] = None
    tgt: Optional[str] = None
    samples: Optional[str] = None

    def merge(self, other: "MethodInfo") -> "MethodInfo":
        """Non-None fields from 'other' override None fields here."""
        return MethodInfo(
            op_code=other.op_code or self.op_code,
            ds=other.ds or self.ds,
            tgt=other.tgt or self.tgt,
            samples=other.samples or self.samples,
        )

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


class Env:
    """Static environment for name and self.attr substitution across methods."""
    def __init__(self) -> None:
        self.locals: Dict[str, ast.AST] = {}       # name -> AST expression
        self.self_attrs: Dict[str, ast.AST] = {}   # self.attr -> AST expression

    def clone(self) -> "Env":
        cp = Env()
        cp.locals.update(self.locals)
        cp.self_attrs.update(self.self_attrs)
        return cp


class _Substituter(ast.NodeTransformer):
    """Replace Name and self.attr nodes with env-provided AST expressions."""
    def __init__(self, env: Env) -> None:
        self.env = env

    def visit_Name(self, node: ast.Name) -> ast.AST:
        repl = self.env.locals.get(node.id)
        return ast.copy_location(repl, node) if repl is not None else node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == Keys.SELF:
            repl = self.env.self_attrs.get(node.attr)
            if repl is not None:
                return ast.copy_location(repl, node)
        return node


def _subst(node: ast.AST, env: Env) -> ast.AST:
    """Apply env-based substitution to an AST node."""
    return _Substituter(env).visit(ast.fix_missing_locations(node))


class TestInfoExtractor(ast.NodeVisitor):
    """
    Extract op_code / ds / tgt / samples for each test* method.

    Key features:
    - Implicitly run setUp once (with default params) before each test (assume_setup=True).
    - Propagate Name/self.attr across helpers (self.* calls) with argument binding (static, no execution).
    - Split samples into ds/tgt when it's a list of dicts with keys {'text','target'}.
    - Merge precedence: class-level defaults < setUp < test (test has highest priority).
    - Results only contain test* methods.

    Integration with utils.parse_class:
    - class_attr_seed is built from extract_class_attr_paths(file, include_containers=True, include_source_fallback=True)
      unless you pass a custom seed. It may include Python literals (lists/dicts/strings) or source strings (e.g. "TestMapper()").
    - literal_eval_with_self is used to turn ds/tgt/samples/op_code into pure literals when possible.
    """

    def __init__(
        self,
        source_code: str,
        file_path: Optional[str] = None,
        class_attr_seed: Optional[Dict[str, Any]] = None,
        assume_setup: bool = True,
    ) -> None:
        try:
            self.tree = ast.parse(source_code)
        except SyntaxError as e:
            raise ValueError(f"Invalid source code syntax: {e}") from e

        # class_name -> { method_name -> FunctionDef }
        self._classes: Dict[str, Dict[str, ast.FunctionDef]] = {}
        # Only test* methods in results
        self.results: Dict[str, Dict[str, Optional[str]]] = {}

        self.file_path = Path(file_path) if file_path else None
        self.assume_setup = assume_setup

        # Seed from class-level attributes (literals and/or source strings)
        if class_attr_seed is not None:
            self.class_attr_seed: Dict[str, Any] = dict(class_attr_seed)
        elif self.file_path:
            # include containers + fallback to source for non-literals (e.g., "TestMapper()")
            try:
                self.class_attr_seed = dict(
                    extract_class_attr_paths(
                        self.file_path,
                        include_containers=True,
                        include_source_fallback=True,
                    )
                )
            except Exception:
                self.class_attr_seed = {}
        else:
            self.class_attr_seed = {}

    def extract(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Entry point."""
        self.visit(self.tree)
        for _, methods in self._classes.items():
            self._process_class(methods)
        return self.results

    # ---------------- Collect methods per class ----------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        methods: Dict[str, ast.FunctionDef] = {}
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods[item.name] = item
        self._classes.setdefault(node.name, {}).update(methods)

    # ---------------- Per-class processing ----------------

    def _process_class(self, methods: Dict[str, ast.FunctionDef]) -> None:
        setup_fn = methods.get(Keys.SETUP)

        # Class-level defaults (lowest priority) from seed
        class_defaults = self._class_defaults_from_seed(self.class_attr_seed)

        for name, fn in methods.items():
            if not name.startswith(Keys.TEST_PREFIX):
                continue

            env = Env()
            # Preheat self.xxx with string-valued class attrs (paths etc.), but skip 'op'
            for k, v in self.class_attr_seed.items():
                if k == Keys.OP:
                    continue
                if isinstance(v, str):
                    env.self_attrs[k] = ast.Constant(value=v)

            # 1) Implicit setUp once with default parameters
            setup_info = MethodInfo()
            if self.assume_setup and setup_fn:
                setup_info = self._run_setup(setup_fn, methods, env)

            # 2) Extract from test body + helpers
            test_info = self._extract_with_env(fn, methods, env, callstack=set())

            # Merge: class_defaults < setUp < test
            info = class_defaults.merge(setup_info).merge(test_info)

            # 3) Literalization: try to turn fields into pure literals (repr), else keep code string
            info.op_code = self._finalize_literal(info.op_code, env)
            info.ds = self._finalize_literal(info.ds, env)
            info.tgt = self._finalize_literal(info.tgt, env)
            info.samples = self._finalize_literal(info.samples, env)

            self.results[name] = info.to_dict()

    def _run_setup(self, setup_fn: ast.FunctionDef, methods: Dict[str, ast.FunctionDef], env: Env) -> MethodInfo:
        """Simulate setUp() with default params (ignore 'self')."""
        params = [a.arg for a in setup_fn.args.args]
        if params and params[0] == Keys.SELF:
            params = params[1:]

        defaults = list(setup_fn.args.defaults or [])
        if defaults:
            tail = params[-len(defaults):]
            for p, d in zip(tail, defaults):
                env.locals[p] = _subst(d, env)

        # Execute setUp body in same env so self.* persists
        return self._extract_with_env(setup_fn, methods, env, callstack={Keys.SETUP})

    # ---------------- Build class-level defaults from seed ----------------

    def _class_defaults_from_seed(self, seed: Dict[str, Any]) -> MethodInfo:
        """Turn class-level ds_list/tgt_list/samples/op from seed into defaults."""
        info = MethodInfo()

        # op may be a source string like "TestMapper()" or a literal string
        if Keys.OP in seed and isinstance(seed[Keys.OP], str):
            info.op_code = seed[Keys.OP]

        # samples: if it's a Python list, split into ds/tgt; else keep as repr
        if Keys.SAMPLES in seed:
            val = seed[Keys.SAMPLES]
            if isinstance(val, list):
                ds, tgt, keep = self._split_samples_py(val)
                if ds is not None:
                    info.ds = info.ds or repr(ds)
                if tgt is not None:
                    info.tgt = info.tgt or repr(tgt)
                if ds is None and tgt is None and keep is not None:
                    info.samples = info.samples or repr(keep)
            elif isinstance(val, (str, dict, tuple)):
                info.samples = info.samples or repr(val)

        # ds_list / tgt_list: prefer Python literals if present, else source strings
        if Keys.DS_LIST in seed:
            val = seed[Keys.DS_LIST]
            info.ds = info.ds or (repr(val) if not isinstance(val, str) else val)
        if Keys.TGT_LIST in seed:
            val = seed[Keys.TGT_LIST]
            info.tgt = info.tgt or (repr(val) if not isinstance(val, str) else val)

        return info

    # ---------------- Core extraction with env/propagation ----------------

    def _extract_with_env(
        self,
        node: ast.FunctionDef,
        methods: Dict[str, ast.FunctionDef],
        env: Env,
        callstack: set,
    ) -> MethodInfo:
        """Sequentially walk statements, update env, and merge extracted info."""
        info = MethodInfo()

        for stmt in node.body:
            # Assignments
            if isinstance(stmt, ast.Assign) and stmt.targets:
                target = stmt.targets[0]
                value = _subst(stmt.value, env)

                # name = expr
                if isinstance(target, ast.Name):
                    env.locals[target.id] = value

                    if target.id == Keys.SAMPLES:
                        ds_code, tgt_code, samples_code = self._split_samples(value)
                        if ds_code:
                            info.ds = info.ds or ds_code
                        if tgt_code:
                            info.tgt = info.tgt or tgt_code
                        if ds_code is None and tgt_code is None:
                            info.samples = samples_code or info.samples

                    elif target.id == Keys.DS_LIST:
                        code = self._safe_unparse(value)
                        if code:
                            info.ds = info.ds or code

                    elif target.id == Keys.TGT_LIST:
                        code = self._safe_unparse(value)
                        if code:
                            info.tgt = info.tgt or code

                    elif target.id == Keys.OP:
                        code = self._safe_unparse(value)
                        if code:
                            info.op_code = info.op_code or code

                # self.attr = expr
                elif (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == Keys.SELF
                ):
                    env.self_attrs[target.attr] = value
                    if target.attr == Keys.OP:
                        code = self._safe_unparse(value)
                        if code:
                            info.op_code = info.op_code or code

                # RHS contains a self.helper(...) call
                if isinstance(stmt.value, ast.Call):
                    child_info = self._maybe_recurse_into_self_call(stmt.value, methods, env, callstack)
                    if child_info:
                        info = info.merge(child_info)

            # Top-level self.helper(...) call (Expr)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                child_info = self._maybe_recurse_into_self_call(stmt.value, methods, env, callstack)
                if child_info:
                    info = info.merge(child_info)

        return info

    def _maybe_recurse_into_self_call(
        self,
        call: ast.Call,
        methods: Dict[str, ast.FunctionDef],
        env: Env,
        callstack: set,
    ) -> Optional[MethodInfo]:
        """If call is self.helper(...), bind args, recurse, propagate self.*, return collected info."""
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == Keys.SELF
        ):
            helper_name = call.func.attr
            if helper_name.startswith(Keys.TEST_PREFIX):
                return None

            callee = methods.get(helper_name)
            if callee and helper_name not in callstack:
                child_env = env.clone()
                bindings = self._bind_call_args(callee, call, env)
                child_env.locals.update(bindings)

                child_stack = set(callstack)
                child_stack.add(helper_name)

                child_info = self._extract_with_env(callee, methods, child_env, child_stack)
                # Propagate self.* updates back to caller env
                env.self_attrs.update(child_env.self_attrs)
                return child_info
        return None

    # ---------------- Argument binding for helpers ----------------

    def _bind_call_args(self, func: ast.FunctionDef, call: ast.Call, parent_env: Env) -> Dict[str, ast.AST]:
        """Bind positional/keyword args at call site to callee parameter names (ignore 'self')."""
        params = [a.arg for a in func.args.args]
        if params and params[0] == Keys.SELF:
            params = params[1:]

        bound: Dict[str, ast.AST] = {}

        # Defaults -> trailing params
        defs = list(func.args.defaults or [])
        if defs:
            tail = params[-len(defs):]
            for p, d in zip(tail, defs):
                bound[p] = _subst(d, parent_env)

        # Positional
        for i, arg in enumerate(call.args[:len(params)]):
            bound[params[i]] = _subst(arg, parent_env)

        # Keywords
        for kw in call.keywords or []:
            if kw.arg in params:
                bound[kw.arg] = _subst(kw.value, parent_env)

        return bound

    # ---------------- samples splitting (AST and Python) ----------------

    def _split_samples(self, node: ast.AST) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        AST-level split: if samples is a list of dicts with 'text'/'target' keys,
        return (ds_code, tgt_code, None); otherwise keep original as code string.
        """
        if isinstance(node, ast.List):
            return self._split_list_samples(node)
        return None, None, self._safe_unparse(node)

    def _split_list_samples(self, node: ast.List) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not node.elts:
            return self._safe_unparse(node), None, None

        first = node.elts[0]
        if not isinstance(first, ast.Dict):
            return self._safe_unparse(node), None, None

        if not self._is_text_target_dict(first):
            return None, None, self._safe_unparse(node)

        try:
            text_vals: List[ast.AST] = []
            target_vals: List[ast.AST] = []
            for el in node.elts:
                if not isinstance(el, ast.Dict) or not self._is_text_target_dict(el):
                    return None, None, self._safe_unparse(node)
                mapping = self._dict_strkey_to_value(el)
                if mapping is None:
                    return None, None, self._safe_unparse(node)
                text_vals.append(mapping[Keys.TEXT])
                target_vals.append(mapping[Keys.TARGET])

            ds_ast = ast.List(elts=text_vals, ctx=ast.Load())
            tgt_ast = ast.List(elts=target_vals, ctx=ast.Load())
            return self._safe_unparse(ds_ast), self._safe_unparse(tgt_ast), None
        except Exception:
            return None, None, self._safe_unparse(node)

    def _split_samples_py(
        self, value: List[Any]
    ) -> Tuple[Optional[List[Any]], Optional[List[Any]], Optional[List[Any]]]:
        """
        Python-level split for class seed: if it's a list of {'text','target'} dicts,
        return (ds_list, tgt_list, None); otherwise keep original list.
        """
        if not value:
            return value, None, None
        first = value[0]
        if not isinstance(first, dict):
            return value, None, None

        for el in value:
            if not (isinstance(el, dict) and set(el.keys()) == {Keys.TEXT, Keys.TARGET}):
                return None, None, value

        ds = [el[Keys.TEXT] for el in value]
        tgt = [el[Keys.TARGET] for el in value]
        return ds, tgt, None

    def _is_text_target_dict(self, node: ast.Dict) -> bool:
        keys = self._dict_str_keys(node)
        return keys is not None and len(keys) == 2 and set(keys) == {Keys.TEXT, Keys.TARGET}

    def _dict_str_keys(self, node: ast.Dict) -> Optional[List[str]]:
        out: List[str] = []
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                out.append(k.value)
            else:
                return None
        return out

    def _dict_strkey_to_value(self, node: ast.Dict) -> Optional[Dict[str, ast.AST]]:
        keys = self._dict_str_keys(node)
        if keys is None:
            return None
        return dict(zip(keys, node.values))

    # ---------------- Utilities ----------------

    def _finalize_literal(self, code: Optional[str], env: Env) -> Optional[str]:
        """
        Try to turn a code string into a pure literal using literal_eval_with_self.
        If conversion fails or no code, return original code string.
        """
        if not code:
            return code

        # Build self.xxx -> string map for replacement
        attr_map: Dict[str, str] = {}
        for k, v in env.self_attrs.items():
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                attr_map[k] = v.value

        try:
            val = literal_eval_with_self(code, attr_map)
            return repr(val) if val is not None else code
        except Exception:
            return code

    def _safe_unparse(self, node: Optional[ast.AST]) -> Optional[str]:
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return None


# ---------------- Convenience APIs ----------------

def extract_test_info_from_source(
    source_code: str,
    *,
    file_path: Optional[str] = None,
    class_attr_seed: Optional[Dict[str, Any]] = None,
    assume_setup: bool = True,
) -> Dict[str, Dict[str, Optional[str]]]:
    extractor = TestInfoExtractor(
        source_code,
        file_path=file_path,
        class_attr_seed=class_attr_seed,
        assume_setup=assume_setup,
    )
    return extractor.extract()


def extract_test_info_from_path(
    py_path: str,
    *,
    class_attr_seed: Optional[Dict[str, Any]] = None,
    assume_setup: bool = True,
) -> Dict[str, Dict[str, Optional[str]]]:
    p = Path(py_path)
    src = p.read_text(encoding="utf-8")
    extractor = TestInfoExtractor(
        src,
        file_path=str(p),
        class_attr_seed=class_attr_seed,
        assume_setup=assume_setup,
    )
    return extractor.extract()
