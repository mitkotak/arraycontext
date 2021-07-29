"""
.. currentmodule:: arraycontext
.. autoclass:: PyOpenCLArrayContext
"""
__copyright__ = """
Copyright (C) 2020-1 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from functools import partial, reduce
import operator

from arraycontext.fake_numpy import \
        BaseFakeNumpyNamespace, BaseFakeNumpyLinalgNamespace
from arraycontext.container.traversal import (
        rec_multimap_array_container, rec_map_array_container,
        rec_map_reduce_array_container,
        )

try:
    import pycuda.gpuarray as gpuarray  a
except ImportError:
    pass


# {{{ fake numpy

class PyCUDAFakeNumpyNamespace(BaseFakeNumpyNamespace):
    def _get_fake_numpy_linalg_namespace(self):
        return _PyCUDAFakeNumpyLinalgNamespace(self._array_context)

    # {{{ comparisons

    # FIXME: This should be documentation, not a comment.
    # These are here mainly because some arrays may choose to interpret
    # equality comparison as a binary predicate of structural identity,
    # i.e. more like "are you two equal", and not like numpy semantics.
    # These operations provide access to numpy-style comparisons in that
    # case.

    def equal(self, x, y):
        return rec_multimap_array_container(operator.eq, x, y)

    def not_equal(self, x, y):
        return rec_multimap_array_container(operator.ne, x, y)

    def greater(self, x, y):
        return rec_multimap_array_container(operator.gt, x, y)

    def greater_equal(self, x, y):
        return rec_multimap_array_container(operator.ge, x, y)

    def less(self, x, y):
        return rec_multimap_array_container(operator.lt, x, y)

    def less_equal(self, x, y):
        return rec_multimap_array_container(operator.le, x, y)

    # }}}

    def ones_like(self, ary):
        def _ones_like(subary):
            ones = self._array_context.empty_like(subary)
            ones.fill(1)
            return ones

        return self._new_like(ary, _ones_like)

    def maximum(self, x, y):
        return rec_multimap_array_container(gpuarray.maximum,x, y)

    def minimum(self, x, y):
        return rec_multimap_array_container(gpuarray.minimum,x, y)

    def where(self, criterion, then, else_):
        return rec_multimap_array_container(gpuarray.where, criterion, then, else_)

    def sum(self, a, dtype=None):
        def _gpuarray_sum(ary):
            if dtype not in [ary.dtype, None]:
                raise NotImplementedError

            return gpuarray.sum(ary)

        return rec_map_reduce_array_container(sum, _gpuarray_sum, a)

    def min(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, gpuarray.minimum), gpuarray.amin, a)

    def max(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, gpuarray.maximum), gpuarray.amax, a)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: gpuarray.stack(arrays=args, axis=axis,
                    self._array_context.allocator),
                *arrays)

    def reshape(self, a, newshape):
        return gpuarray.reshape(a, newshape)

    def concatenate(self, arrays, axis=0):
        return  gpuarray.concatenate(
            arrays, axis,
            self._array_context.allocator
        )

    def ravel(self, a, order="C"):
        def _rec_ravel(a):
            if order in "FCA":
                return a.reshape(-1, order=order)

            elif order == "K":
                raise NotImplementedError("PyCUDAArrayContext.np.ravel not "
                                          "implemented for 'order=K'")
            else:
                raise ValueError("`order` can be one of 'F', 'C', 'A' or 'K'. "
                                 f"(got {order})")

        return rec_map_array_container(_rec_ravel, a)

# }}}


# {{{ fake np.linalg

class _PyCUDAFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    pass

# }}}


# vim: foldmethod=marker
