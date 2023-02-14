import ctypes
from typing import Sequence, Dict, Union

import llvmlite.ir
from llvmlite.ir import IRBuilder, Value as IRValue

from dispyatcher import Handle, Type


class ResourceHandle(Handle):
    """
    Creates a handle that acquires and releases a resource automatically.

    It is effectively the handle version of a `with` block. It takes three handles: a constructor, an inner handle, and
    a destructor. The inner handle can, optionally, be passed the value produced by the constructor.
    """
    __constructor: Handle
    __destructor: Handle
    __index: Union[int, None]
    __inner: Handle

    def __init__(self, inner: Handle, index: Union[int, None], constructor: Handle, destructor: Handle):
        """
        Create a new resource management handle

        The constructor can take arguments and the resulting handle will contain those arguments. Where the arguments
        will appear depends on whether `index` is supplied.

        If `index` is supplied, the constructor arguments will
        appear at the position where the resource is used. That is:
        ```
        with allocate_resource(x, y) as r:
            inner_handle(a, b, r, c)
        ```
        The resulting handle will expect arguments `(a, b, x, y, c)`.

        If the index is `None`, then constructor arguments are prepended to the inner handle's arguments:
        ```
        with allocate_resource(x, y):
            inner_handle(a, b, c)
        ```
        The resulting handle will expect arguments `(x, y, a, b, c)`.
        :param inner: the handle to run while the resource is allocated
        :param index: if a number, the index in the inner handle that will use the resource; if `None`, the resource
        will not be passed to the inner handle
        :param constructor: a handle that allocates the constructor. Arguments to this handled will be added to the
        inner handle's arguments for the overall arguments to the handle
        :param destructor: a handle that deallocates the value. This handle must take exactly one argument with the same
        type as the return type of the constructor and no return value.
        """
        super().__init__()
        self.__constructor = constructor
        self.__destructor = destructor
        self.__index = index
        self.__inner = inner

        (_, inner_args) = inner.function_type()
        (ctor_ret, ctor_args) = constructor.function_type()
        (dtor_ret, dtor_args) = destructor.function_type()
        assert isinstance(dtor_ret.machine_type(), llvmlite.ir.VoidType), "Destructor cannot return a value"
        assert len(dtor_args) == 1, "Destructor must take single argument."
        assert dtor_args[0] == ctor_ret,\
            f"Expected destructor {dtor_args[0]} to match constructor return type {ctor_ret}"
        if index is not None:
            assert 0 <= index < len(inner_args), "Index is out of range"
            assert inner_args[index] == ctor_ret, f"Expected {ctor_ret} at {index}, but got {inner_args[index]}."

        constructor.register(self)
        destructor.register(self)
        inner.register(self)

    def function_type(self) -> (Type, Sequence[Type]):
        (inner_ret, inner_args) = self.__inner.function_type()
        (_, ctor_args) = self.__constructor.function_type()
        args = []
        if self.__index is None:
            args.extend(ctor_args)
            args.extend(inner_args)
        else:
            args.extend(inner_args[0:self.__index])
            args.extend(ctor_args)
            args.extend(inner_args[self.__index + 1:])
        return inner_ret, args

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        num_ctor_args = len(self.__constructor.function_type()[1])
        if self.__index is None:
            ctor_args = args[0:num_ctor_args]
        else:
            ctor_args = args[self.__index:self.__index + num_ctor_args]
        value = self.__constructor.generate_ir(builder, ctor_args, global_addresses)
        if self.__index is None:
            inner_args = args[num_ctor_args:]
        else:
            inner_args = []
            inner_args.extend(args[0:self.__index])
            inner_args.append(value)
            inner_args.extend(args[self.__index + 1:])
        result = self.__inner.generate_ir(builder, inner_args, global_addresses)
        self.__destructor.generate_ir(builder, (value,), global_addresses)
        return result

    def __str__(self) -> str:
        return f"Call {self.__inner} With {self.__constructor} at {self.__index} Then {self.__destructor}"



