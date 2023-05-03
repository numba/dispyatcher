import ctypes
import unittest

import llvmlite.ir
import llvmlite.ir

import dispyatcher
import dispyatcher.general
from dispyatcher import CallSite, IdentityHandle, IgnoreArgumentsHandle
from dispyatcher.general import SimpleConstantHandle
from dispyatcher.cpython import TupleUnpackingDispatcher, PY_OBJECT_TYPE, with_gil, PythonControlFlowType


class RepackTests(unittest.TestCase):

    def test_tuple_unpack_int(self):
        i32 = dispyatcher.general.MachineType(llvmlite.ir.IntType(32))
        tud = TupleUnpackingDispatcher(
            IgnoreArgumentsHandle(
                SimpleConstantHandle(
                    i32, -1), 0, PY_OBJECT_TYPE), 0)
        tud.append(IdentityHandle(i32), None)
        callsite = CallSite(with_gil(tud), PythonControlFlowType())
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
        callsite = CallSite(with_gil(tud), PythonControlFlowType())
        self.assertEqual(callsite.cfunc(ctypes.py_object((7.4,))), 7.4)
        self.assertEqual(callsite.cfunc(ctypes.py_object(7)), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object((7, 7))), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object("foo")), -1)
        self.assertEqual(callsite.cfunc(ctypes.py_object((7,))), 7.0)


class PythonFlowTests(unittest.TestCase):

    def test_dictionary_size(self):
        handle = dispyatcher.cpython.PY_DICT_SIZE
        callsite = CallSite(with_gil(handle), PythonControlFlowType())
        self.assertEqual(callsite.cfunc(ctypes.py_object({})), 0)
        self.assertEqual(callsite.cfunc(ctypes.py_object({"a": 1})), 1)
        self.assertEqual(callsite.cfunc(ctypes.py_object({"a": 1, "b": 1})), 2)

    def test_dictionary_non_null(self):
        handle = dispyatcher.cpython.PY_DICT_GET_ITEM + dispyatcher.cpython.IncrementReference +\
                 dispyatcher.cpython.NullToNone
        callsite = CallSite(with_gil(handle), PythonControlFlowType())
        self.assertEqual(callsite.cfunc(ctypes.py_object({"a": 1}), ctypes.py_object("a")), 1)
        self.assertEqual(callsite.cfunc(ctypes.py_object({"a": 1}), ctypes.py_object("b")), None)

    def test_dictionary_throw_if_null(self):
        handle = dispyatcher.cpython.PY_DICT_GET_ITEM + dispyatcher.cpython.ThrowIfNull +\
                 dispyatcher.cpython.IncrementReference
        callsite = CallSite(with_gil(handle), PythonControlFlowType())
        self.assertEqual(callsite.cfunc(ctypes.py_object({"a": 1}), ctypes.py_object("a")), 1)
        with self.assertRaises(ValueError):
            callsite.cfunc(ctypes.py_object({"a": 1}), ctypes.py_object("b"))
