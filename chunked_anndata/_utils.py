from typing import Tuple, Union, Optional, get_origin, get_args
import functools 
import warnings
import inspect
import collections


def deprecated(*, ymd: Tuple[int] = None, optional_message: str = None):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn("Function {} is going to be deprecated".format(func.__name__) + " after %d-%d-%d." % ymd if ymd else '.',
                        category=DeprecationWarning,
                        stacklevel = 2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)
        return wrapper 
    return decorate

def is_instance_of_type(value, expected_type):
    """Check if a value is an instance of the expected type, accounting for Optional."""
    # Handle Optional (which is Union[T, None])
    if get_origin(expected_type) is Union:
        allowed_types = get_args(expected_type)
        return any(is_instance_of_type(value, t) for t in allowed_types)
    elif get_origin(expected_type) is Optional:
        allowed_types = get_args(expected_type)
        return value is None or any([is_instance_of_type(value, t) for t in allowed_types])
    elif  get_origin(expected_type) is collections.abc.Mapping:
        return isinstance(value, dict) and all(
            is_instance_of_type(k, expected_type.__args__[0])
            and is_instance_of_type(v, expected_type.__args__[1])
            for k, v in value.items()
        )
    elif get_origin(expected_type) is list:
        return isinstance(value, list) and all(
            is_instance_of_type(v, expected_type.__args__[0])
            for v in value
        )
    
    return isinstance(value, expected_type)


def typed(types: dict = None, *, optional_message: str = None):
    def decorate(func):
        arg_names = inspect.getfullargspec(func)[0]
        @functools.wraps(func)
        def wrapper(*args,**kwargs):
            if types:
                for name, arg in zip(arg_names, args):
                    if name in types and not is_instance_of_type(arg, types[name]):
                        raise TypeError("Argument {} must be {}".format(name, types[name]))
                for name, arg in kwargs.items():
                    if name in types and not is_instance_of_type(arg, types[name]):
                        raise TypeError("Argument {} must be {}".format(name, types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate