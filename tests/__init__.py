import ctypes
import dispyatcher
import dispyatcher.general
import llvmlite.ir
import unittest


class SimpleTests(unittest.TestCase):

    def test_echo_i32(self):
        callsite = dispyatcher.CallSite(dispyatcher.Identity(dispyatcher.general.MachineType(llvmlite.ir.IntType(32)))
                                        + dispyatcher.Clone)
        self.assertEqual(callsite(ctypes.c_int32(42)), 42)

    def test_echo_f64(self):
        callsite = dispyatcher.CallSite(dispyatcher.Identity(dispyatcher.general.MachineType(llvmlite.ir.DoubleType()))
                                        + dispyatcher.Clone)
        self.assertEqual(callsite(ctypes.c_double(1.618)), 1.618)

    def test_echo_int_narrow(self):
        callsite = dispyatcher.CallSite(dispyatcher.Identity(dispyatcher.general.MachineType(llvmlite.ir.IntType(8)))
                                        // dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
                                        / (dispyatcher.general.MachineType(llvmlite.ir.IntType(32)),)
                                        + dispyatcher.Clone)
        self.assertEqual(callsite(ctypes.c_int32(1024)), 0)

    def test_echo_int_float(self):
        callsite = dispyatcher.CallSite(dispyatcher.Identity(dispyatcher.general.MachineType(llvmlite.ir.DoubleType()))
                                        // dispyatcher.general.MachineType(llvmlite.ir.DoubleType())
                                        / (dispyatcher.general.MachineType(llvmlite.ir.IntType(32)),)
                                        + dispyatcher.Clone)
        self.assertEqual(callsite(ctypes.c_int32(1024)), 1024.0)

    def test_echo_float_int(self):
        callsite = dispyatcher.CallSite(dispyatcher.Identity(dispyatcher.general.MachineType(llvmlite.ir.IntType(32)))
                                        // dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
                                        / (dispyatcher.general.MachineType(llvmlite.ir.DoubleType()),)
                                        + dispyatcher.Clone)
        self.assertEqual(callsite(ctypes.c_double(102.1)), 102)

    def test_echo_float_double(self):
        callsite = dispyatcher.CallSite(dispyatcher.Identity(dispyatcher.general.MachineType(llvmlite.ir.FloatType()))
                                        // dispyatcher.general.MachineType(llvmlite.ir.DoubleType())
                                        / (dispyatcher.general.MachineType(llvmlite.ir.FloatType()),)
                                        + dispyatcher.Clone)
        self.assertNotEqual(callsite(ctypes.c_float(102.1)), 102.1)

    def test_echo_ignore_start(self):
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        callsite = dispyatcher.CallSite(dispyatcher.IgnoreArguments(dispyatcher.Identity(i32), 0, i32)
                                        + dispyatcher.Clone)
        self.assertEqual(callsite(ctypes.c_int32(42), ctypes.c_int32(47)), 47)

    def test_echo_ignore_end(self):
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        callsite = dispyatcher.CallSite(dispyatcher.IgnoreArguments(dispyatcher.Identity(i32), 1, i32))
        self.assertEqual(callsite(ctypes.c_int32(42), ctypes.c_int32(47)), 42)

    def test_constant_int(self):
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        callsite = dispyatcher.CallSite(
            dispyatcher.general.SimpleConstant(i32, 7, transfer=dispyatcher.ReturnManagement.TRANSFER))
        self.assertEqual(callsite(), 7)


class CallSiteTest(unittest.TestCase):

    def test_update_callsite(self):
        i8 = dispyatcher.general.MachineType(llvmlite.ir.IntType(8))
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        inner_callsite = dispyatcher.CallSite(dispyatcher.Identity(i32))
        callsite = dispyatcher.CallSite(dispyatcher.IgnoreArguments(inner_callsite, 1, i32) + dispyatcher.Clone)
        self.assertEqual(callsite(ctypes.c_int32(1024), ctypes.c_int32(47)), 1024)
        inner_callsite.handle = dispyatcher.Identity(i8) // i32 / (i32,)
        self.assertEqual(callsite(ctypes.c_int32(1025), ctypes.c_int32(47)), 1)
