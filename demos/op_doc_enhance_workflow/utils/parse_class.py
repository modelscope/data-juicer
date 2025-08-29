#!/usr/bin/env python3
import ast
from pathlib import Path
from typing import Any, Dict, Optional

from data_juicer.utils.mm_utils import SpecialTokens

# Environment for evaluating expressions
env = {
    "SpecialTokens": SpecialTokens,
}


def _eval_path_expr(node: ast.AST, env: Dict[str, Any]) -> Optional[str]:
    """Evaluate path expressions supporting strings, __file__, variables, and os.path calls"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        return None
    if isinstance(node, ast.Attribute):
        # os.path attributes don't return values directly, usually appear in Call.func
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        func_attr = node.func
        # Support os.path.<method> calls
        if isinstance(func_attr.value, ast.Attribute) and isinstance(func_attr.value.value, ast.Name):
            if func_attr.value.value.id == "os" and func_attr.value.attr == "path":
                fname = func_attr.attr
                args = [_eval_path_expr(a, env) for a in node.args]
                if any(a is None for a in args):
                    return None
                import os as _os

                # Handle different os.path methods
                if fname == "join":
                    return _os.path.normpath(_os.path.join(*args))
                if fname == "dirname":
                    return _os.path.dirname(args[0])
                if fname == "realpath":
                    return _os.path.realpath(args[0])
                if fname == "abspath":
                    return _os.path.abspath(args[0])
                if fname == "normpath":
                    return _os.path.normpath(args[0])
        # Nested calls will be processed by outer layers
    return None


def extract_class_attr_paths(pyfile: Path) -> Dict[str, str]:
    """
    Extract path constants from test class attributes (e.g. data_path, img1_path).
    Supports combinations of: os.path methods + __file__ + predefined variables.
    """
    src = pyfile.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(pyfile))
    env = {"__file__": str(pyfile)}
    import os as _os

    env["os"] = _os  # For attribute existence check (not direct evaluation)

    # Process class definitions
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    name = stmt.targets[0].id
                    val = _eval_path_expr(stmt.value, env)
                    if val:
                        env[name] = val
    return {k: v for k, v in env.items() if isinstance(v, str) and k not in ("__file__", "os")}


def _eval_expr(node: ast.AST, env: Dict[str, Any]) -> Optional[Any]:
    """Evaluate general expressions including constants, names, and attributes"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.Attribute):
        obj = _eval_expr(node.value, env)
        if obj is not None and hasattr(obj, node.attr):
            return getattr(obj, node.attr)
    return None


class ASTTransformer(ast.NodeTransformer):
    """AST transformer to replace self.xxx and f-string variables with constants"""

    def __init__(self, attr_map: Dict[str, str], env: Dict[str, Any]):
        self.attr_map = attr_map  # Map for replacing self.xxx
        self.env = env  # Environment for replacing f-string variables

    def visit_Attribute(self, node: ast.Attribute):
        """Replace self.xxx with string constants"""
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr in self.attr_map:
                return ast.copy_location(ast.Constant(value=self.attr_map[node.attr]), node)
        return self.generic_visit(node)

    def visit_FormattedValue(self, node: ast.FormattedValue):
        """Replace f-string formatted values with evaluated constants"""
        node = self.generic_visit(node)

        expr_value = _eval_expr(node.value, self.env)
        if expr_value is None:
            return node

        text = str(expr_value)
        return ast.copy_location(ast.Constant(value=text), node)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        """Replace f-strings with joined constant strings"""
        node = self.generic_visit(node)

        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                return node

        joined = "".join(parts)
        return ast.copy_location(ast.Constant(value=joined), node)


class TestCaseExtractor(ast.NodeVisitor):
    """AST visitor to extract test case information from test files"""

    def __init__(self, source_code):
        self.source_code = source_code
        self.methods = {}
        self.current_method = None

    def visit_FunctionDef(self, node):
        """Visit function definition nodes to track current test method"""
        self.current_method = node.name
        self.methods[self.current_method] = {
            "op_code": None,
            "ds": None,
            "tgt": None,
            "samples": None,
        }
        self.generic_visit(node)
        self.current_method = None

    def visit_Assign(self, node):
        """Visit assignment nodes to extract test data variables"""
        if not self.current_method:
            return
        target = node.targets[0]
        raw_code = ast.unparse(node.value)
        if isinstance(target, ast.Name):
            # Extract different types of test data
            if target.id == "ds_list":
                self.methods[self.current_method]["ds"] = raw_code
            elif target.id == "tgt_list":
                self.methods[self.current_method]["tgt"] = raw_code
            elif target.id == "op":
                self.methods[self.current_method]["op_code"] = raw_code
            elif target.id == "samples":
                self.methods[self.current_method]["samples"] = raw_code


def literal_eval_with_self(raw_code: str, attr_map: Dict[str, str]) -> Any:
    """
    Replace self.xxx with string constants in expression, then literal_eval.
    raw_code should be an expression (like list/dict literals).
    """
    expr = ast.parse(raw_code, mode="eval")
    transformer = ASTTransformer(attr_map, env)
    new_expr = transformer.visit(expr)
    ast.fix_missing_locations(new_expr)
    try:
        return ast.literal_eval(new_expr.body)
    except Exception:
        return None
