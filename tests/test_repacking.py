import ctypes
import unittest
from typing import Sequence, Union, Any, Dict

import llvmlite.ir
import llvmlite.ir
from llvmlite.ir import IRBuilder, Value as IRValue, Block

import dispyatcher
import dispyatcher.general
from dispyatcher import CallSite, IdentityHandle, IgnoreArgumentsHandle, Type
from dispyatcher.general import SimpleConstantHandle
from dispyatcher.repacking import Repacker, RepackingDispatcher


class RepackTests(unittest.TestCase):

    def test_exact_ints(self):
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        repacking = SummingRepackingHandle(IgnoreArgumentsHandle(SimpleConstantHandle(i32, 42), 0, i32, i32, i32), 1)
        repacking.append(IgnoreArgumentsHandle(IdentityHandle(i32), 0, i32), 7)
        callsite = CallSite(repacking)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0)), 42)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(3)), 42)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(100), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(3)),
                         42)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(0), ctypes.c_int32(900), ctypes.c_int32(0), ctypes.c_int32(3)),
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

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue], failure_block: Block,
                    global_addresses: Dict[str, ctypes.c_char_p]) -> Sequence[IRValue]:
        value = args[0]
        for arg in args[1:]:
            value = builder.add(value, arg)
        condition = builder.icmp_signed('>', value, llvmlite.ir.Constant(self.__type, self.__hint))
        ok_block = builder.append_basic_block("summing_ok")
        builder.cbranch(condition, ok_block, failure_block)
        builder.position_at_start(ok_block)
        return builder.mul(value, llvmlite.ir.Constant(self.__type, 10)),


class SummingRepackingHandle(RepackingDispatcher):
    def __init__(self, handle: dispyatcher.Handle, common_input: int):
        super().__init__(handle, common_input)

    def _find_repack(self, input_args: Sequence[Type], output_args: Sequence[Type], hint: Any)\
            -> Union[Repacker, None]:
        if len(output_args) == 1 and isinstance(output_args[0].machine_type(), llvmlite.ir.IntType) and\
                all(arg.machine_type() == output_args[0].machine_type() for arg in input_args):
            return SummingRepacker(len(input_args), int(hint), output_args[0].machine_type())
