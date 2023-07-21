import itertools
import llvmlite.ir
from llvmlite.ir import Value as IRValue
from typing import Generic, Tuple, Sequence, Union

from dispyatcher import F, Handle, Type, ReturnManagement, ArgumentManagement, TemporaryValue, Pointer
from dispyatcher.general import UncheckedArray


class ToStackPointer(Handle[F], Generic[F]):
    """
    Creates a stack allocation for a value and provides a pointer to it.

    The allocation can take ownership of the value and the value will be freed when the pointer is dropped.
    """
    __type: Type
    __transfer: ReturnManagement

    def __init__(self, ty: Type, transfer: ReturnManagement = ReturnManagement.TRANSFER):
        """
        Create a new stack allocation handle for a particular type.

        :param ty: the type the allocation contains; the result will be a pointer to this type
        :param transfer: whether to move the value into the allocation (``TRANSFER``) or copy it and rely on the
            original value for memory management (``BORROW``).
        """
        super().__init__()
        self.__type = ty
        self.__transfer = transfer

    def __str__(self) -> str:
        return f"ToStackPointer[{self.__transfer.name}] → {self.__type}"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        ty = self.__type.machine_type()
        (arg, ) = args
        alloc = flow.builder.alloca(ty)
        flow.builder.store(arg, alloc)
        return alloc

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return (self.__type,
                ArgumentManagement.TRANSFER_CAPTURE_PARENTS
                if self.__transfer == ReturnManagement.TRANSFER
                else ArgumentManagement.BORROW_CAPTURE_PARENTS),

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return Pointer(self.__type), self.__transfer


class CollectIntoArray(Handle[F], Generic[F]):
    """
    Stores a arguments into a stack-allocated fixed-length array.

    This is meant for producing arrays like ``argv`` from individual values
    """
    __type: Type
    __count: int
    __null_terminated: bool

    def __init__(self, ty: Type, count: int, null_terminated: bool = False):
        """
        Creates a new stack-allocated array collector handle.

        :param ty: the type of the elements in the array
        :param count: the number of items to collect
        :param null_terminated: if true, the array will be allocated with an extra slot at the end that will have a null
            or zero value; otherwise, the array will only be filled with the arguments.
        """
        super().__init__()
        assert count > 0, "Array length must be positive number"
        self.__type = ty
        self.__count = count
        self.__null_terminated = null_terminated

    def __str__(self) -> str:
        return f"CollectIntoArray({self.__type} * {self.__count} {'+null' if self.__null_terminated else ''})"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        i32 = llvmlite.ir.IntType(32)
        array = flow.builder.alloca(llvmlite.ir.ArrayType(self.__type.machine_type(),
                                                          self.__count + self.__null_terminated))
        for idx, arg in enumerate(args):
            flow.builder.store(arg, flow.builder.gep(array, [i32(0), i32(idx)]))
        if self.__null_terminated:
            flow.builder.store(self.__type.machine_type()(None), flow.builder.gep(array, [i32(0), i32(self.__count)]))
        return flow.builder.bitcast(array, self.__type.machine_type().as_pointer())

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return list(itertools.repeat((self.__type, ArgumentManagement.BORROW_CAPTURE), self.__count))

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return UncheckedArray(self.__type), ReturnManagement.BORROW
