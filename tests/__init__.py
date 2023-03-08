import ctypes
import dispyatcher
import dispyatcher.general
import llvmlite.ir
import unittest


class SimpleTests(unittest.TestCase):

    def test_echo_i32(self):
        callsite = dispyatcher.CallSite(dispyatcher.IdentityHandle(
            dispyatcher.general.MachineType(llvmlite.ir.IntType(32))))
        self.assertEqual(callsite.cfunc(ctypes.c_int32(42)), 42)

    def test_echo_f64(self):
        callsite = dispyatcher.CallSite(dispyatcher.IdentityHandle(
            dispyatcher.general.MachineType(llvmlite.ir.DoubleType())))
        self.assertEqual(callsite.cfunc(ctypes.c_double(1.618)), 1.618)

    def test_echo_int_narrow(self):
        callsite = dispyatcher.CallSite(dispyatcher.IdentityHandle(
            dispyatcher.general.MachineType(llvmlite.ir.IntType(8)))
                                        .cast(dispyatcher.general.MachineType(llvmlite.ir.IntType(32)),
                                              dispyatcher.general.MachineType(llvmlite.ir.IntType(32))))
        self.assertEqual(callsite.cfunc(ctypes.c_int32(1024)), 0)

    def test_echo_int_float(self):
        callsite = dispyatcher.CallSite(dispyatcher.IdentityHandle(
            dispyatcher.general.MachineType(llvmlite.ir.DoubleType()))
                                        .cast(dispyatcher.general.MachineType(llvmlite.ir.DoubleType()),
                                              dispyatcher.general.MachineType(llvmlite.ir.IntType(32))))
        self.assertEqual(callsite.cfunc(ctypes.c_int32(1024)), 1024.0)

    def test_echo_float_int(self):
        callsite = dispyatcher.CallSite(dispyatcher.IdentityHandle(
            dispyatcher.general.MachineType(llvmlite.ir.IntType(32)))
                                        .cast(dispyatcher.general.MachineType(llvmlite.ir.IntType(32)),
                                              dispyatcher.general.MachineType(llvmlite.ir.DoubleType())))
        self.assertEqual(callsite.cfunc(ctypes.c_double(102.1)), 102)

    def test_echo_float_double(self):
        callsite = dispyatcher.CallSite(dispyatcher.IdentityHandle(
            dispyatcher.general.MachineType(llvmlite.ir.FloatType()))
                                        .cast(dispyatcher.general.MachineType(llvmlite.ir.DoubleType()),
                                              dispyatcher.general.MachineType(llvmlite.ir.FloatType())))
        self.assertNotEqual(callsite.cfunc(ctypes.c_float(102.1)), 102.1)

    def test_echo_drop_start(self):
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        callsite = dispyatcher.CallSite(dispyatcher.IgnoreArgumentsHandle(dispyatcher.IdentityHandle(i32), 0, i32))
        self.assertEqual(callsite.cfunc(ctypes.c_int32(42), ctypes.c_int32(47)), 47)

    def test_echo_drop_end(self):
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        callsite = dispyatcher.CallSite(dispyatcher.IgnoreArgumentsHandle(dispyatcher.IdentityHandle(i32), 1, i32))
        self.assertEqual(callsite.cfunc(ctypes.c_int32(42), ctypes.c_int32(47)), 42)

    def test_constant_int(self):
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        callsite = dispyatcher.CallSite(dispyatcher.general.SimpleConstantHandle(i32, 7))
        self.assertEqual(callsite.cfunc(), 7)


class CallSiteTest(unittest.TestCase):

    def test_update_callsite(self):
        i8 = dispyatcher.general.MachineType(llvmlite.ir.IntType(8))
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        inner_callsite = dispyatcher.CallSite(dispyatcher.IdentityHandle(i32))
        callsite = dispyatcher.CallSite(dispyatcher.IgnoreArgumentsHandle(inner_callsite, 1, i32))
        self.assertEqual(callsite.cfunc(ctypes.c_int32(1024), ctypes.c_int32(47)), 1024)
        inner_callsite.handle = dispyatcher.IdentityHandle(i8).cast(i32, i32)
        self.assertEqual(callsite.cfunc(ctypes.c_int32(1025), ctypes.c_int32(47)), 1)
