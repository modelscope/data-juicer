import typing
import inspect

try:
    from pydantic.types import PositiveInt, PositiveFloat
    from jsonargparse.typing import ClosedUnitInterval

    pydantic_types = {
        PositiveInt: ("int", 1, None),
        PositiveFloat: ("float", 0.000001, None),
        ClosedUnitInterval: ("float", 0.0, 1.0),
    }
except ImportError:
    pydantic_types = {}


class TypeCategory:
    def __init__(self, type_obj, type_str, converter, default, min_val=None, max_val=None):
        self.type_obj = type_obj
        self.type_str = type_str
        self.converter = converter
        self.default = default
        self.min = min_val
        self.max = max_val

    def convert(self, val):
        return self.converter(val)

    @property
    def is_optional(self):
        import typing

        origin = typing.get_origin(self.type_obj)
        args = typing.get_args(self.type_obj)
        return origin is typing.Union and type(None) in args


class TypeAnnotationParser:
    """Parser for type annotations with support for complex types and constraints.

    Handles conversion between type annotations and string representations,
    supports pydantic types, and provides type validation and conversion utilities.
    """

    def __init__(self, extra_types: dict = None):
        """Initialize the parser with basic and optional extra type mappings.

        :param extra_types: Additional type mappings to extend the default set
        :type extra_types: dict, optional
        """
        self.type_map = {
            "int": int,
            "str": str,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "None": type(None),
            "List": list,
            "Dict": dict,
            "Tuple": tuple,
        }
        if extra_types:
            self.type_map.update(extra_types)
        for t in pydantic_types:
            self.type_map[t.__name__] = t

    def type_to_string(self, annotation) -> str:
        """Convert type annotation to string representation.

        :param annotation: Type annotation to convert
        :return: String representation of the type
        :rtype: str
        """
        if annotation in (str, int, float, bool):
            return annotation.__name__

        if annotation in pydantic_types:
            return pydantic_types[annotation][0]

        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin is typing.Union:
            if len(args) == 2 and type(None) in args:
                non_none = args[0] if args[1] is type(None) else args[1]
                return f"Optional[{self.type_to_string(non_none)}]"
            return f'Union[{", ".join(self.type_to_string(a) for a in args)}]'
        elif origin in (list, typing.List):
            return f"List[{self.type_to_string(args[0])}]" if args else "List"
        elif origin in (dict, typing.Dict):
            return f"Dict[{self.type_to_string(args[0])},{self.type_to_string(args[1])}]" if len(args) == 2 else "Dict"
        elif origin in (tuple, typing.Tuple):
            return f'Tuple[{", ".join(self.type_to_string(a) for a in args)}]' if args else "Tuple"
        elif origin is typing.Annotated:
            return self.type_to_string(args[0])
        elif origin is None and annotation is type(None):
            return "None"
        # fallback
        return str(annotation).replace("typing.", "")

    def str_to_type(self, type_str: str):
        """Convert string representation back to type object.

        :param type_str: String representation of type
        :type type_str: str
        :return: Type object
        """
        ns = dict(vars(typing))
        ns.update(self.type_map)
        return eval(type_str, {"__builtins__": None}, ns)

    def parse_annotation(self, annotation):
        """Parse type annotation and extract constraints.

        :param annotation: Type annotation to parse
        :return: Tuple of (type_string, min_value, max_value)
        :rtype: tuple
        """
        if annotation == inspect.Parameter.empty or annotation is None:
            return None, None, None

        # Handle pydantic special types (e.g., PositiveInt)
        if annotation in pydantic_types:
            return pydantic_types[annotation]

        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)

        if origin is typing.Annotated:
            # Format: Annotated[base_type, *metadata]
            base_type = args[0]
            type_str = self.type_to_string(base_type)
            min_val, max_val = self._extract_constraints(args[1:])
            return type_str, min_val, max_val
        elif origin is typing.Union:
            # Optional or Union types
            if len(args) == 2 and type(None) in args:
                base_type = args[0] if args[1] is type(None) else args[1]
                base_str, minv, maxv = self.parse_annotation(base_type)
                return f"Optional[{base_str}]", minv, maxv
            # Union types
            return self.type_to_string(annotation), None, None
        else:
            return self.type_to_string(annotation), None, None

    def _extract_constraints(self, metadata):
        """Extract min/max constraints from annotation metadata.

        :param metadata: Metadata from Annotated type
        :return: Tuple of (min_value, max_value)
        :rtype: tuple
        """
        min_val, max_val = None, None
        for meta_item in metadata:
            for name in ["ge", "gt", "le", "lt"]:
                if hasattr(meta_item, name) and getattr(meta_item, name) is not None:
                    v = getattr(meta_item, name)
                    if name in ("ge", "gt"):
                        min_val = v
                    if name in ("le", "lt"):
                        max_val = v
            if hasattr(meta_item, "metadata"):
                sub_min, sub_max = self._extract_constraints(meta_item.metadata)
                if sub_min is not None:
                    min_val = sub_min
                if sub_max is not None:
                    max_val = sub_max
        return min_val, max_val

    def get_default_value(self, typ):
        """Infer default value for given type, supports complex types like tuple[int, int].

        :param typ: Type to get default value for
        :return: Default value for the type
        """
        # Support string type input
        if isinstance(typ, str):
            typ = self.str_to_type(typ)
        origin = typing.get_origin(typ)
        args = typing.get_args(typ)
        base = {str: "", int: 0, float: 0.0, bool: False, list: [], dict: {}, tuple: ()}
        if typ in base:
            return base[typ]
        if origin in (list, typing.List):
            return []
        if origin in (dict, typing.Dict):
            return {}
        # Handle specific tuple types
        if origin in (tuple, typing.Tuple):
            # Empty tuple for typing.Tuple with no args
            if not args:
                return ()
            # Handle ellipsis notation like Tuple[int, ...]
            if len(args) == 2 and args[1] is Ellipsis:
                # Generate single element for demonstration
                default_x = self.get_default_value(args[0])
                return (default_x,)
            # Regular tuple with specific types
            return tuple(self.get_default_value(a) for a in args)
        # Optional[X] returns None
        if origin is typing.Union:
            non_none_list = [a for a in args if a is not type(None)]
            return self.get_default_value(non_none_list[0])
        return None

    def converter(self, type_annotation):
        """Create converter function for string values to target type (supports Optional, List recursively).

        :param type_annotation: Type annotation to create converter for
        :return: Converter function
        """
        t = type_annotation
        if isinstance(type_annotation, str):
            t = self.str_to_type(type_annotation)
        origin = typing.get_origin(t)
        args = typing.get_args(t)
        # Basic primitive types
        if t in (int, float, str, bool, list, dict, tuple):
            return t
        # Optional types
        if origin is typing.Union and type(None) in args:
            inner = args[0] if args[1] is type(None) else args[1]
            inner_conv = self.converter(inner)

            def opt_fn(x):
                if x is None:
                    return None
                return inner_conv(x)

            return opt_fn
        # List types
        if origin in (list, typing.List):
            conv = self.converter(args[0])

            def fn(x):
                if not hasattr(x, "__iter__") or isinstance(x, (str, bytes)):
                    x = [x]
                return [conv(i) for i in x]

            return fn
        # Tuple types
        if origin in (tuple, typing.Tuple):
            convs = [self.converter(a) for a in args]

            def fn(x):
                if not hasattr(x, "__iter__") or isinstance(x, (str, bytes)):
                    x = [x]
                x = list(x)
                if len(x) != len(convs):
                    raise ValueError("Tuple size mismatch")
                return tuple(f(i) for f, i in zip(convs, x))

            return fn
        # Union types
        if origin is typing.Union:
            convs = [self.converter(a) for a in args if a is not type(None)]

            def try_union(x):
                for f in convs:
                    try:
                        return f(x)
                    except Exception:
                        continue
                if None in args:
                    return None
                raise ValueError("Cannot coerce to Union types")

            return try_union
        # Default fallback
        return str

    def get_type_category(self, typ):
        """Create TypeCategory object for a type annotation.

        :param typ: Type annotation
        :return: TypeCategory instance with type information and utilities
        :rtype: TypeCategory
        """
        if isinstance(typ, str):
            typ = self.str_to_type(typ)
        type_str, min_val, max_val = self.parse_annotation(typ)
        return TypeCategory(typ, type_str, self.converter(typ), self.get_default_value(typ), min_val, max_val)
