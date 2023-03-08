import ctypes
import unittest

import llvmlite.ir
import llvmlite.ir

import dispyatcher
import dispyatcher.general
from dispyatcher import CallSite, IdentityHandle, IgnoreArgumentsHandle
from dispyatcher.general import SimpleConstantHandle
from dispyatcher.cpython import TupleUnpackingDispatcher, PY_OBJECT_TYPE, with_gil


class RepackTests(unittest.TestCase):

    def test_tuple_unpack_int(self):
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        tud = TupleUnpackingDispatcher(
            IgnoreArgumentsHandle(
                SimpleConstantHandle(
                    i32, -1), 0, PY_OBJECT_TYPE), 0)
        tud.append(IdentityHandle(i32), None)
        callsite = CallSite(with_gil(tud))
        self.assertEqual(callsite.cfunc(ctypes.py_object((7,))), 7)
        self.assertEqual(callsite.cfunc(ctypes.py_object(7)), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object((7, 7))), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object("foo")), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object((7.5,))), -1)

    def test_tuple_unpack_double(self):
        dbl = dispyatcher.general.MachineType(llvmlite.ir.DoubleType())
        tud = TupleUnpackingDispatcher(
            IgnoreArgumentsHandle(
                SimpleConstantHandle(
                    dbl, -1), 0, PY_OBJECT_TYPE), 0)
        tud.append(IdentityHandle(dbl), None)
        callsite = CallSite(with_gil(tud))
        self.assertEqual(callsite.cfunc(ctypes.py_object((7.4,))), 7.4)
        self.assertEqual(callsite.cfunc(ctypes.py_object(7)), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object((7, 7))), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object("foo")), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object((7,))), 7.0)
