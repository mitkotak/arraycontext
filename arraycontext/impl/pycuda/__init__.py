"""
.. currentmodule:: arraycontext
.. autoclass:: PyCUDAArrayContext
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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

from warnings import warn
from typing import Dict, List, Sequence, Optional, Union, TYPE_CHECKING

import numpy as np

from pytools.tag import Tag

from arraycontext.context import ArrayContext


if TYPE_CHECKING:
    import pycuda


# {{{ PyCUDAArrayContext

class PyCUDAArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :class:`pycuda.gpuarray.GPUArray` instances
    for its base array class.

    .. attribute:: allocator

        A PyCUDA memory allocator. Can also be `None` (default) or `False` to
        use the default allocator.

    .. automethod:: __init__
    """

    def __init__(self, allocator=None):
        import pycuda
        super().__init__()
        if allocator == None:
            self.allocator =  pycuda.driver.mem_alloc
            from warnings import warn
            warn("Allocator is None")
        else:
            self.allocator = allocator

    def _get_fake_numpy_namespace(self):
        from arraycontext.impl.pycuda.fake_numpy import PyCUDAFakeNumpyNamespace
        return PyCUDAFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def empty(self, shape, dtype):
        import pycuda.gpuarray as gpuarray
        return gpuarray.empty(shape=shape, dtype=dtype,
                allocator=self.allocator)

    def zeros(self, shape, dtype):
        import pycuda.gpuarray as gpuarray
        return gpuarray.zeros(shape=shape, dtype=dtype,
                allocator=self.allocator)

    def from_numpy(self, array: np.ndarray):
        import pycuda.gpuarray as gpuarray
        return gpuarray.to_gpu(array, allocator=self.allocator)

    def to_numpy(self, array):
        import pycuda.gpuarray as gpuarray
        return array.get()

    def call_loopy(self, t_unit, **kwargs):
        raise NotImplementedError('Waiting for loopy to be more capable')

    def freeze(self, array):
        return array

    def thaw(self, array):
        return array

    # }}}

    def clone(self):
        return type(self)(self.allocator)

    def tag(self, array):
        return array

    def tag_axis(self, array):
        return array

    @property
    def permits_inplace_modification(self):
        return True

# }}}

# vim: foldmethod=marker
