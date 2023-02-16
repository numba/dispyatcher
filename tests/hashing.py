import ctypes
import unittest

import llvmlite.ir
import llvmlite.ir
from llvmlite.ir import IRBuilder, Value as IRValue

import dispyatcher
from dispyatcher import CallSite, IdentityHandle, SimpleConstantHandle, IgnoreArgumentsHandle, Type
from dispyatcher.hashing import HashValueDispatcher, HashValueGuard


class HashingTests(unittest.TestCase):

    def test_exact_ints(self):
        i32 = dispyatcher.MachineType(llvmlite.ir.IntType(32))
        hashing = HashValueDispatcher(IdentityHandle(i32), ExactIntegerGuard())
        hashing.insert((7,), IgnoreArgumentsHandle(SimpleConstantHandle(i32, 42), 0, i32))
        callsite = CallSite(hashing)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(0)), 0)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(6)), 6)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(7)), 42)

    def test_lowbit_ints(self):
        i32 = dispyatcher.MachineType(llvmlite.ir.IntType(32))
        hashing = HashValueDispatcher(IdentityHandle(i32), LowBitIntegerGuard())
        hashing.insert((7,), IgnoreArgumentsHandle(SimpleConstantHandle(i32, 42), 0, i32))
        hashing.insert((1,), IgnoreArgumentsHandle(SimpleConstantHandle(i32, 7), 0, i32))
        callsite = CallSite(hashing)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(0)), 0)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(1)), 7)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(6)), 6)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(3)), 3)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(7)), 42)

    def test_lowbit_ints_2args(self):
        i32 = dispyatcher.MachineType(llvmlite.ir.IntType(32))
        hashing = HashValueDispatcher(IgnoreArgumentsHandle(IdentityHandle(i32), 1, i32),
                                      LowBitIntegerGuard(),
                                      LowBitIntegerGuard())
        hashing.insert((1, 1), IgnoreArgumentsHandle(SimpleConstantHandle(i32, 42), 0, i32, i32))
        hashing.insert((3, 4), IgnoreArgumentsHandle(SimpleConstantHandle(i32, 7), 0, i32, i32))
        callsite = CallSite(hashing)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(0), ctypes.c_int32(0)), 0)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(1), ctypes.c_int32(1)), 42)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(0), ctypes.c_int32(1)), 0)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(1), ctypes.c_int32(0)), 1)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(3), ctypes.c_int32(4)), 7)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(3), ctypes.c_int32(3)), 3)


class ExactIntegerGuard(HashValueGuard):

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "Exact Integer"

    def compatible(self, ty: Type) -> bool:
        return ty.machine_type() == llvmlite.ir.IntType(32)

    def compute_hash(self, value) -> int:
        return int(value)

    def generate_hash_ir(self, builder: IRBuilder, arg: IRValue) -> IRValue:
        return arg

    def generate_check_ir(self, value, builder: IRBuilder, arg: IRValue) -> IRValue:
        return builder.icmp_signed('==', arg, llvmlite.ir.Constant(llvmlite.ir.IntType(32), value))


class LowBitIntegerGuard(HashValueGuard):

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "Low-bit Integer"

    def compatible(self, ty: Type) -> bool:
        return ty.machine_type() == llvmlite.ir.IntType(32)

    def compute_hash(self, value) -> int:
        return int(value) & 1

    def generate_hash_ir(self, builder: IRBuilder, arg: IRValue) -> IRValue:
        return builder.and_(arg, llvmlite.ir.Constant(llvmlite.ir.IntType(32), 1))

    def generate_check_ir(self, value, builder: IRBuilder, arg: IRValue) -> IRValue:
        return builder.icmp_signed('==', arg, llvmlite.ir.Constant(llvmlite.ir.IntType(32), value))


