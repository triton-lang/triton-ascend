import inspect


class ArgumentBinder:
    """Python argument matching without invoking the native specializer."""

    def __init__(self, signature):
        parameters = tuple(signature.parameters.values())
        if any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters):
            raise ValueError("unsupported JIT parameter kind")
        scope = {}
        arguments = []
        for i, p in enumerate(parameters):
            item = p.name
            if p.default is not inspect.Parameter.empty:
                scope[f"_default_{i}"] = p.default
                item += f"=_default_{i}"
            arguments.append(item)
        options_name = "_reuse_options"
        while options_name in signature.parameters:
            options_name += "_"
        arguments.append("**" + options_name)
        entries = ", ".join(f"{p.name!r}: {p.name}" for p in parameters)
        # Names come from inspect.Parameter; defaults are namespace values, never code.
        source = f"def dynamic_func({', '.join(arguments)}):\n    return {{{entries}}}, {options_name}\n"
        exec(source, scope)  # noqa: S102
        self.bind = scope["dynamic_func"]

    def __call__(self, *args, **kwargs):
        return self.bind(*args, **kwargs)
