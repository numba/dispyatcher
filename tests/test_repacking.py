import ctypes
import unittest
from typing import Sequence, Union, Any, Tuple, Set

import llvmlite.ir
from llvmlite.ir import Value as IRValue

import dispyatcher
import dispyatcher.general
from dispyatcher import CallSite, Identity, IgnoreArguments, Type
from dispyatcher.general import SimpleConstant
from dispyatcher.repacking import Repacker, RepackingDispatcher, RepackingState


class RepackTests(unittest.TestCase):

    def test_exact_ints(self):
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        repacking = SummingRepackingHandle(IgnoreArguments(SimpleConstant(i32, 42) + dispyatcher.Clone,
                                                           0,
                                                           i32,
                                                           i32,
                                                           i32,
                                                           capture=dispyatcher.IgnoreCapture.CAPTURE_PARENTS), 1)
        repacking.append(IgnoreArguments(Identity(i32) + dispyatcher.Clone,
                                         0,
                                         i32,
                                         capture=dispyatcher.IgnoreCapture.CAPTURE_PARENTS), 7)
        callsite = CallSite(repacking)
        self.assertEqual(callsite(ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0)), 42)
        self.assertEqual(callsite(ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(3)), 42)
        self.assertEqual(callsite(ctypes.c_int32(100), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(3)),
                         42)
        self.assertEqual(callsite(ctypes.c_int32(0), ctypes.c_int32(900), ctypes.c_int32(0), ctypes.c_int32(3)),
                         9000)


class SummingRepacker(Repacker):
    __count: int
    __hint: int
    __type: llvmlite.ir.IntType

    def __init__(self, count, hint, ty) -> None:
        super().__init__()
        self.__count = count
        self.__hint = hint
        self.__type = ty

    def __str__(self) -> str:
        return "Summing"

    def input_count(self) -> int:
        return self.__count

    def output_count(self) -> int:
        return 1

    def generate_ir(self, state: RepackingState, args: Sequence[IRValue]) -> Sequence[Tuple[IRValue, Set[int]]]:
        value = args[0]
        for arg in args[1:]:
            value = state.flow.builder.add(value, arg)
        state.alternate_on_bool(state.flow.builder.icmp_signed('>',
                                                               value,
                                                               llvmlite.ir.Constant(self.__type, self.__hint)))
        return (state.flow.builder.mul(value, llvmlite.ir.Constant(self.__type, 10)), set(range(self.__count))),


class SummingRepackingHandle(RepackingDispatcher):
    def __init__(self, handle: dispyatcher.Handle, common_input: int):
        super().__init__(handle, common_input)

    def _find_repack(self,
                     input_args: Sequence[Tuple[Type, dispyatcher.ArgumentManagement]],
                     output_args: Sequence[Tuple[Type, dispyatcher.ArgumentManagement]],
                     hint: Any)\
            -> Union[Repacker, None]:
        if len(output_args) == 1 and isinstance(output_args[0][0].machine_type(), llvmlite.ir.IntType) and\
                all(arg_ty.machine_type() == output_args[0][0].machine_type() and
                    arg_mgmt in (dispyatcher.ArgumentManagement.BORROW_TRANSIENT,
                                 dispyatcher.ArgumentManagement.BORROW_CAPTURE,
                                 dispyatcher.ArgumentManagement.BORROW_CAPTURE_PARENTS)
                    for arg_ty, arg_mgmt in input_args):
            return SummingRepacker(len(input_args), int(hint), output_args[0][0].machine_type())
