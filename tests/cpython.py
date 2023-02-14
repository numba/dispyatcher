import ctypes
import unittest

import llvmlite.ir
import llvmlite.ir

import dispyatcher
from dispyatcher import CallSite, IdentityHandle, SimpleConstantHandle, DropArgumentsHandle
from dispyatcher.cpython import TupleUnpackingDispatcher, PY_OBJECT_TYPE, with_gil


class RepackTests(unittest.TestCase):

    def test_tuple_unpack_int(self):
        i32 = dispyatcher.MachineType(llvmlite.ir.IntType(32))
        tud = TupleUnpackingDispatcher(DropArgumentsHandle(SimpleConstantHandle(i32, -1), 0, PY_OBJECT_TYPE), 0)
        tud.append(IdentityHandle(i32), None)
        callsite = CallSite(with_gil(tud))
        self.assertEqual(callsite.cfunc(ctypes.py_object((7,))), 7)
        self.assertEqual(callsite.cfunc(ctypes.py_object(7)), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object((7, 7))), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object("foo")), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object((7.5,))), -1)

    def test_tuple_unpack_double(self):
        dbl = dispyatcher.MachineType(llvmlite.ir.DoubleType())
        tud = TupleUnpackingDispatcher(DropArgumentsHandle(SimpleConstantHandle(dbl, -1), 0, PY_OBJECT_TYPE), 0)
        tud.append(IdentityHandle(dbl), None)
        callsite = CallSite(with_gil(tud))
        self.assertEqual(callsite.cfunc(ctypes.py_object((7.4,))), 7.4)
        self.assertEqual(callsite.cfunc(ctypes.py_object(7)), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object((7, 7))), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object("foo")), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object((7,))), 7.0)
