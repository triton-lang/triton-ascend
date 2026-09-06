import inspect
from functools import wraps
from warnings import warn

_DEPRECATED_MESSAGE_ATTR = "_deprecated_message"


def _deprecated(fn_name=None, replacement=None):

    def decorator(fn):
        name = fn_name or f"triton.language.extra.cann.extension.{fn.__name__}"
        message = (f"{name} is deprecated and will be removed in the next release"
                   f"{f'; use {replacement} instead.' if replacement else '.'}")

        if inspect.isclass(fn):
            setattr(fn, _DEPRECATED_MESSAGE_ATTR, message)
            fn.__doc__ = f"{fn.__doc__ or ''}\n\n.. warning::\n   {message}"
            return fn

        @wraps(fn)
        def wrapper(*args, **kwargs):
            warn(message, FutureWarning, stacklevel=2)
            return fn(*args, **kwargs)

        wrapper.__doc__ = f"{fn.__doc__ or ''}\n\n.. warning::\n   {message}"
        return wrapper

    return decorator
