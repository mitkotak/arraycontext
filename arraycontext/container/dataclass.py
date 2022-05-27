# mypy: disallow-untyped-defs

"""
.. currentmodule:: arraycontext
.. autofunction:: dataclass_array_container
"""


__copyright__ = """
Copyright (C) 2020-1 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Union, get_args
try:
    # NOTE: only available in python >= 3.8
    from typing import get_origin
except ImportError:
    from typing_extensions import get_origin

from dataclasses import fields
from arraycontext.container import is_array_container_type


# {{{ dataclass containers

def dataclass_array_container(cls: type) -> type:
    """A class decorator that makes the class to which it is applied an
    :class:`ArrayContainer` by registering appropriate implementations of
    :func:`serialize_container` and :func:`deserialize_container`.
    *cls* must be a :func:`~dataclasses.dataclass`.

    Attributes that are not array containers are allowed. In order to decide
    whether an attribute is an array container, the declared attribute type
    is checked by the criteria from :func:`is_array_container_type`.
    """
    from dataclasses import is_dataclass, Field
    assert is_dataclass(cls)

    def is_array_field(f: Field) -> bool:
        from arraycontext import Array

        origin = get_origin(f.type)
        if origin is Union:
            if not all(
                    arg is Array or is_array_container_type(arg)
                    for arg in get_args(f.type)):
                raise TypeError(
                        f"Field '{f.name}' union contains non-array container "
                        "arguments. All arguments must be array containers.")
            else:
                return True

        if __debug__:
            if not f.init:
                raise ValueError(
                        f"'init=False' field not allowed: '{f.name}'")

            if isinstance(f.type, str):
                raise TypeError(
                        f"string annotation on field '{f.name}' not supported")

            from typing import _SpecialForm
            if isinstance(f.type, _SpecialForm):
                # NOTE: anything except a Union is not allowed
                raise TypeError(
                        f"typing annotation not supported on field '{f.name}': "
                        f"'{f.type!r}'")

            if not isinstance(f.type, type):
                raise TypeError(
                        f"field '{f.name}' not an instance of 'type': "
                        f"'{f.type!r}'")

        return f.type is Array or is_array_container_type(f.type)

    from pytools import partition
    array_fields, non_array_fields = partition(is_array_field, fields(cls))

    if not array_fields:
        raise ValueError(f"'{cls}' must have fields with array container type "
                "in order to use the 'dataclass_array_container' decorator")

    serialize_expr = ", ".join(
            f"({f.name!r}, ary.{f.name})" for f in array_fields)
    template_kwargs = ", ".join(
            f"{f.name}=template.{f.name}" for f in non_array_fields)

    lower_cls_name = cls.__name__.lower()

    serialize_init_code = ", ".join(f"{f.name!r}: f'{{instance_name}}.{f.name}'"
            for f in array_fields)
    deserialize_init_code = ", ".join([
            f"{f.name}={{args[{f.name!r}]}}" for f in array_fields
            ] + [
            f"{f.name}={{template_instance_name}}.{f.name}"
            for f in non_array_fields
            ])

    from pytools.codegen import remove_common_indentation
    serialize_code = remove_common_indentation(f"""
        from typing import Any, Iterable, Tuple
        from arraycontext import serialize_container, deserialize_container

        @serialize_container.register(cls)
        def _serialize_{lower_cls_name}(ary: cls) -> Iterable[Tuple[Any, Any]]:
            return ({serialize_expr},)

        @deserialize_container.register(cls)
        def _deserialize_{lower_cls_name}(
                template: cls, iterable: Iterable[Tuple[Any, Any]]) -> cls:
            return cls(**dict(iterable), {template_kwargs})

        # support for with_container_arithmetic

        def _serialize_init_arrays_code_{lower_cls_name}(cls, instance_name):
            return {{
                {serialize_init_code}
                }}
        cls._serialize_init_arrays_code = classmethod(
            _serialize_init_arrays_code_{lower_cls_name})

        def _deserialize_init_arrays_code_{lower_cls_name}(
                cls, template_instance_name, args):
            return f"{deserialize_init_code}"

        cls._deserialize_init_arrays_code = classmethod(
            _deserialize_init_arrays_code_{lower_cls_name})
        """)

    exec_dict = {"cls": cls, "_MODULE_SOURCE_CODE": serialize_code}
    exec(compile(serialize_code, f"<container serialization for {cls.__name__}>",
        "exec"), exec_dict)

    return cls

# }}}

# vim: foldmethod=marker
