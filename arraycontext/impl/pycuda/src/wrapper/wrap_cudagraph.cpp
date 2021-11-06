#include <cuda.hpp>
#include <cudagraph.hpp>

#include <utility>
#include <numeric>
#include <algorithm>

#include "tools.hpp"
#include "wrap_helpers.hpp"
#include <boost/python/stl_iterator.hpp>

#if CUDAPP_CUDA_VERSION < 1010
#error PyCuda only works with CUDA 1.1 or newer.
#endif

