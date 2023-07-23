/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_blas.h"

#include "tensorflow/compiler/xla/stream_executor/rocm/rocblas_wrapper.h"

#define EIGEN_USE_GPU
#include <assert.h>

#include <complex>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/compiler/xla/stream_executor/platform/dso_loader.h"
#include "tensorflow/compiler/xla/stream_executor/platform/initialize.h"
#include "tensorflow/compiler/xla/stream_executor/platform/logging.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/plugin_registry.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/scratch_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/util/determinism.h"
using tsl::OpDeterminismRequired;

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kRocBlasPlugin);

extern void rocm_Broadcast_fp32(void *stream, float *dst, int dst_stride,
                                int batches, int src_batches, float *src,
                                int size);

template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    const DeviceMemory<T> &a) {
  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type *>(
      GpuMemory(a));
}

template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    const T &a) {
  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type *>(&a);
}
template <class T>
typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    DeviceMemory<T> *a) {
  return reinterpret_cast<
      typename RocBlasTypeConversionHelper<T>::mapped_type *>(
      GpuMemoryMutable(a));
}

static void blas_log(const char *c) {}


static string ToString(rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return "rocblas_status_success";
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    default:
      return absl::StrCat("<invalid rocBLAS status: ", status, ">");
  }
}

bool ROCMBlas::Init() {
  gpu::ScopedActivateExecutorContext sac{parent_};
  rocblas_status ret = wrap::rocblas_create_handle(&blas_);
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to create rocBLAS handle: " << ToString(ret);
    return false;
  }

#if TF_HIPBLASLT
  if (!blas_lt_.Init().ok()) {
    LOG(ERROR) << "Failed to initialize hipblasLt";
    return false;
  }
#endif
  return true;
}

ROCMBlas::ROCMBlas(gpu::GpuExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr)
#if TF_HIPBLASLT
    , 
    blas_lt_(parent)
#endif    
     {}

ROCMBlas::~ROCMBlas() {
  if (blas_ != nullptr) {
    gpu::ScopedActivateExecutorContext sac{parent_};
    wrap::rocblas_destroy_handle(blas_);
  }
}

bool ROCMBlas::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  gpu::ScopedActivateExecutorContext sac{parent_};
  rocblas_status ret =
      wrap::rocblas_set_stream(blas_, AsGpuStreamValue(stream));
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to set stream for rocBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

namespace {

// Helper functions transforming blas arguments into rocBLAS arguments.

rocblas_operation ROCMBlasTranspose(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return rocblas_operation_none;
    case blas::Transpose::kTranspose:
      return rocblas_operation_transpose;
    case blas::Transpose::kConjugateTranspose:
      return rocblas_operation_conjugate_transpose;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

rocblas_fill ROCMBlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return rocblas_fill_upper;
    case blas::UpperLower::kLower:
      return rocblas_fill_lower;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

rocblas_diagonal ROCMBlasDiagonal(blas::Diagonal diag) {
  switch (diag) {
    case blas::Diagonal::kUnit:
      return rocblas_diagonal_unit;
    case blas::Diagonal::kNonUnit:
      return rocblas_diagonal_non_unit;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}

rocblas_side ROCMBlasSide(blas::Side side) {
  switch (side) {
    case blas::Side::kLeft:
      return rocblas_side_left;
    case blas::Side::kRight:
      return rocblas_side_right;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}

}  // namespace

template <typename FuncT, typename... Args>
bool ROCMBlas::DoBlasInternalImpl(FuncT rocblas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  Args... args) {
  absl::MutexLock lock{&mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  gpu::ScopedActivateExecutorContext sac{parent_};

  // set the atomics mode, leaving default to library
  bool allow_atomics = !OpDeterminismRequired();
  rocblas_status ret;
  if (!allow_atomics) {
    ret = wrap::rocblas_set_atomics_mode(blas_, rocblas_atomics_not_allowed);
    if (err_on_failure && ret != rocblas_status_success) {
      LOG(ERROR) << "failed to to set atomics mode before "
                 << rocblas_func.kName << ": " << ToString(ret);
    }
  }

  ret = rocblas_func(blas_, args...);
  if (err_on_failure && ret != rocblas_status_success) {
    LOG(ERROR) << "failed to run ROCBLAS routine " << rocblas_func.kName << ": "
               << ToString(ret);
  }
  return ret == rocblas_status_success;
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  blas_log("DoBlasAxpy");
  return DoBlasInternal(wrap::rocblas_saxpy, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  blas_log("DoBlasAxpy");
  return DoBlasInternal(wrap::rocblas_daxpy, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_caxpy, stream, /* pointer_mode_host = */ true, elem_count,
      complex_cast(alpha), complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_zaxpy, stream, /* pointer_mode_host = */ true, elem_count,
      complex_cast(alpha), complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_scopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_dcopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_ccopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_zcopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  blas_log("DoBlasScal<float>");
  return DoBlasInternal(wrap::rocblas_sscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_dscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_csscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_zdscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_cscal, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(alpha), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_zscal, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(alpha), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_sgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_dgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_cgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  blas_log("DoBlasGemv\n");
  return DoBlasInternal(
      wrap::rocblas_zgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_ssbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, k, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_dsbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, k, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

template <typename T, typename U=T>
U alpha_cast(const void* ptr, blas::DataType dtype, float defval) {
  T val = T(defval);
  if(ptr != nullptr) {
    if(blas::ToDataType<T>::value == dtype)
      val = *(reinterpret_cast<const T*>(ptr));
    else {
      if(dtype == blas::DataType::kHalf)
        val = T(*(reinterpret_cast<const Eigen::half*>(ptr)));
      else if(dtype == blas::DataType::kFloat)
        val = T(*(reinterpret_cast<const float*>(ptr)));
      else if(dtype == blas::DataType::kDouble)
        val = T(*(reinterpret_cast<const double*>(ptr)));
      else if(dtype == blas::DataType::kBF16)
        val = T(*(reinterpret_cast<const Eigen::bfloat16*>(ptr)));
      else if(dtype == blas::DataType::kComplexFloat)
        val = T(reinterpret_cast<const std::complex<float>*>(ptr)->real());
      else if(dtype == blas::DataType::kComplexDouble)
        val = T(reinterpret_cast<const std::complex<double>*>(ptr)->real());
      else
        throw "Unsupported alpha_cast type";
    }
  }
  return reinterpret_cast<const U&>(val);
}

template <typename T, typename U, typename V>
tsl::Status ROCMBlas::DoBlasGemmInternalNonEx(Stream *stream, V fun, const blas::GemmCall& call) {
  blas::Transpose transa = call.transa;
  blas::Transpose transb = call.transb;
  uint64_t m = call.m;
  uint64 n = call.n;
  uint64_t k = call.k;
  blas::DataType dtype = call.dtype_in;
  const DeviceMemoryBase &a = *call.pa;
  int lda = call.lda;
  const DeviceMemoryBase &b = *call.pb;
  int ldb = call.ldb;
  DeviceMemoryBase *c = call.c;
  int ldc = call.ldc;
  blas::ComputePrecision precision = call.precision;

  auto alpha = alpha_cast<T,U>(call.alpha, call.dtype_ab, 1.);
  auto beta = alpha_cast<T,U>(call.beta, call.dtype_ab, 0.);

  return DoBlasInternalStatus(
      fun, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
      &alpha,
      reinterpret_cast<const U*>(a.opaque()), lda,
      reinterpret_cast<const U*>(b.opaque()), ldb,
      &beta,
      reinterpret_cast<U*>(c->opaque()), ldc);
}

tsl::Status ROCMBlas::DoBlasGemmInternalEx(Stream *stream, const blas::GemmCall& call, rocblas_datatype dt) {
  blas::Transpose transa = call.transa;
  blas::Transpose transb = call.transb;
  uint64_t m = call.m;
  uint64 n = call.n;
  uint64_t k = call.k;
  blas::DataType dtype = call.dtype_in;
  const DeviceMemoryBase &a = *call.pa;
  int lda = call.lda;
  const DeviceMemoryBase &b = *call.pb;
  int ldb = call.ldb;
  DeviceMemoryBase *c = call.c;
  int ldc = call.ldc;
  blas::ComputePrecision precision = call.precision;

  auto alpha = alpha_cast<float>(call.alpha, call.dtype_ab, 1);
  auto beta = alpha_cast<float>(call.beta, call.dtype_ab, 0);

  return DoBlasInternalStatus(
      wrap::rocblas_gemm_ex, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb),
      (rocblas_int)m, (rocblas_int)n, (rocblas_int)k, &alpha, a.opaque(),
      dt, lda, b.opaque(), dt,
      ldb, &beta, c->opaque(), dt, ldc, c->opaque(),
      dt, ldc, rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard, 0, 0);
}
/*
tsl::Status ROCMBlas::DoBlasGemmInternalEx2(Stream* stream, const blas::GemmCall& call)
{
#if ROCBLAS_VERSION_MAJOR>3 || (ROCBLAS_VERSION_MAJOR==3 && ROCBLAS_VERSION_MINOR>=1) 
      if(call.context & blas::CallContext::kEnableF8) {
            rocblas_computetype compute_type;
            switch (int(call.context) & 3) {
              case 0:
                compute_type = rocblas_compute_type_f8_f8_f32;
                break;
              case 1:
                compute_type = rocblas_compute_type_bf8_f8_f32;
                break;
              case 2:          
               compute_type = rocblas_compute_type_f8_bf8_f32;
                break;
              case 3:
                return tsl::errors::Internal(absl::StrCat("Unexpected call_context for GEMM: ",
                                                  int(call.context)));
            }
            return DoBlasInternalStatus(
                wrap::rocblas_gemm_ext2, stream,  true,
                ROCMBlasTranspose(transa), ROCMBlasTranspose(transb),
                (rocblas_int)m, (rocblas_int)n, (rocblas_int)k, alpha, a,
                rocblas_datatype_f16_r, lda, b, rocblas_datatype_f16_r,
                ldb, beta, c, rocblas_datatype_f16_r, ldc, c,
                rocblas_datatype_f16_r, ldc, compute_type,
                rocblas_gemm_algo_standard, 0, 0x18);    
        } 
#else
    return tsl::errors::Internal("This rocBLAS has no FP8 support");
#endif
}
*/
tsl::Status ROCMBlas::DoBlasGemm(Stream *stream, const blas::GemmCall& call) {
  if(call.dtype_in != call.dtype_out)
    return tsl::errors::Internal("ROCMBlas::DoBlasGemm does not support mixed data types");
  if(call.batch_count>1)
    return DoBlasGemmStridedBatched(stream, call);
  blas::Transpose transa = call.transa;
  blas::Transpose transb = call.transb;
  uint64_t m = call.m;
  uint64 n = call.n;
  uint64_t k = call.k;
  blas::DataType dtype = call.dtype_in;
  const DeviceMemoryBase &a = *call.pa;
  int lda = call.lda;
  const DeviceMemoryBase &b = *call.pb;
  int ldb = call.ldb;
  DeviceMemoryBase *c = call.c;
  int ldc = call.ldc;
  blas::ComputePrecision precision = call.precision;

  auto falpha = alpha_cast<float>(call.alpha, call.dtype_ab, 1);
  auto fbeta = alpha_cast<float>(call.beta, call.dtype_ab, 0);

  VLOG(1) << absl::StreamFormat(
      "doing rocBLAS GEMM: at=%d bt=%d m=%u n=%u "
      "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, falpha,
      a.opaque(), lda, b.opaque(), ldb, fbeta, c->opaque(), ldc);

  if(!(int(call.context) & int(blas::CallContext::kSet)))
    return tsl::errors::Internal("ROCMBlas::DoBlasGemm did not receive gradient flags");

  if (dtype == blas::DataType::kHalf || dtype == blas::DataType::kFloat) {
    if (transa == blas::Transpose::kNoTranspose) {
      if (lda < static_cast<int64_t>(m)) {
        LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                        "precondition violation";
      }
    } else {
      if (lda < static_cast<int64_t>(k)) {
        LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                     << ") (transpose case); precondition violation";
      }
    }
    if (transb == blas::Transpose::kNoTranspose) {
      if (ldb < static_cast<int64_t>(k)) {
        LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                     << ") (no transpose case); precondition violation";
      }
    } else {
      if (ldb < static_cast<int64_t>(n)) {
        LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                        "precondition violation";
      }
    }
  }

  tsl::StatusOr<bool> maybe_hasXDLOPS = GpuDriver::GetMFMASupport();


#if ROCBLAS_VERSION_MAJOR>3 || (ROCBLAS_VERSION_MAJOR==3 && ROCBLAS_VERSION_MINOR>=1) 
  //if(dtype==blas::DataType::kHalf && (call.context & blas::CallContext::kEnableF8)) {
  if(dtype==blas::DataType::kHalf) {
    return DoBlasGemmInternalEx2(stream, call);
  }
#endif

  if((dtype==blas::DataType::kHalf 
      && maybe_hasXDLOPS.ok() 
      && maybe_hasXDLOPS.value())
      || (dtype==blas::DataType::kBF16)) {
      auto rtype = (dtype==blas::DataType::kBF16) 
          ? rocblas_datatype_bf16_r
          : rocblas_datatype_f16_r; 
      return DoBlasGemmInternalEx(stream, call, rtype);
  }
  // FIXME: review that all of these possibilities are touched by unit tests
  // (esp. with nonnull alpha & beta)
  switch (dtype) {
  case blas::DataType::kHalf: 
    return DoBlasGemmInternalNonEx<Eigen::half, rocblas_half>(stream, wrap::rocblas_hgemm, call);
  case blas::DataType::kFloat:
    return DoBlasGemmInternalNonEx<float,float>(stream, wrap::rocblas_sgemm, call);
  case blas::DataType::kDouble:
    return DoBlasGemmInternalNonEx<double,double>(stream, wrap::rocblas_dgemm, call);
  case blas::DataType::kComplexFloat:
     return DoBlasGemmInternalNonEx<std::complex<float>, rocblas_float_complex>
         (stream, wrap::rocblas_cgemm, call);
  case blas::DataType::kComplexDouble:
     return DoBlasGemmInternalNonEx<std::complex<double>, rocblas_double_complex>
         (stream, wrap::rocblas_zgemm, call);
  default:
    return tsl::errors::Internal("Unsupported datatype for GEMM: ",
                                 blas::DataTypeString(dtype));
  }
}

bool ROCMBlas::GetBlasGemmAlgorithms(
    Stream *stream, std::vector<blas::AlgorithmType> *out_algorithms) {
  // ROCM TODO: properly implement the interface
  return true;
}

struct MemoryCopyOp {
  char *src_ptr;
  char *dst_ptr;
  uint64_t size;
  uint64_t count;
  uint64_t dst_stride;
  uint64_t src_count;
};

// Check whether two Memory Copy Ops can be fold together.
// If it's true, fold it. Otherwise, return false.
static bool MemCopyOpsFold(MemoryCopyOp &y, const MemoryCopyOp &x) {
  bool misaligned = (x.size & 3) ||
                    (reinterpret_cast<uint64_t>(x.dst_ptr) & 3) ||
                    (reinterpret_cast<uint64_t>(x.src_ptr) & 3) ||
                    (reinterpret_cast<uint64_t>(y.dst_ptr) & 3) ||
                    (reinterpret_cast<uint64_t>(y.src_ptr) & 3);

  int64_t dst_step = reinterpret_cast<int64_t>(x.dst_ptr) -
                     reinterpret_cast<int64_t>(y.dst_ptr);

  if (x.src_ptr == y.src_ptr && x.size == y.size &&
      (y.count == 1 || x.dst_ptr == y.dst_ptr + y.count * y.dst_stride) &&
      !misaligned && y.src_count == 1 && !(dst_step & 3)) {
    if (y.count == 1) {
      y.dst_stride = dst_step;
    }
    y.count++;
    return true;
  } else if (x.src_ptr == y.src_ptr + y.size &&
             x.dst_ptr == y.dst_ptr + y.size && y.count == 1 &&
             y.src_count == 1) {
    y.size += x.size;
    return true;
  }
  if (x.src_ptr == y.src_ptr + y.size * y.src_count &&
      x.dst_ptr == y.dst_ptr + y.dst_stride * y.src_count * y.count &&
      x.count == y.count && x.dst_stride == y.dst_stride) {
    y.src_count += x.src_count;
    return true;
  }
  return false;
}

// This copies from source memory: raw_ptrs[i] to target memory:
// device_memory_ptr at the interval of matrix_byte_size, or vice versa.
// The below algorithm tries to minimize the number of memcpy by consolidating
// neighboring memcpy into a single request.
template <typename MAPPED_T>
tsl::Status ReorganizeMemory(Stream *stream,
                             DeviceMemory<MAPPED_T> *device_memory,
                             const std::vector<MAPPED_T *> &raw_ptrs,
                             int batch_count, uint64_t batch_stride,
                             bool gather) {
  if (gather == false) {
    return tsl::Status(absl::StatusCode::kUnimplemented,
                       "gather=false is unsupported");
  }

  assert(batch_count > 0);
  char *device_memory_ptr = static_cast<char *>(device_memory->opaque());
  char* src_ptr = reinterpret_cast<char*>(raw_ptrs[0]);
  char* dst_ptr = device_memory_ptr;
  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);

  std::vector<MemoryCopyOp> mem_copy_ops{
      MemoryCopyOp{src_ptr, dst_ptr, matrix_byte_size, 1, 0, 1}};

  for (int i = 1; i < batch_count; ++i) {
    src_ptr = reinterpret_cast<char *>(raw_ptrs[i]);
    dst_ptr = device_memory_ptr + i * matrix_byte_size;

    MemoryCopyOp x{src_ptr, dst_ptr, matrix_byte_size, 1, 0, 1};
    while (mem_copy_ops.size() > 1 &&
           MemCopyOpsFold(mem_copy_ops[mem_copy_ops.size() - 2],
                          mem_copy_ops.back())) {
      mem_copy_ops.pop_back();
    }
    MemoryCopyOp &op = mem_copy_ops.back();
    if (MemCopyOpsFold(op, x)) {
      continue;
    }
    mem_copy_ops.push_back(x);
  }

  while (mem_copy_ops.size() > 1 &&
         MemCopyOpsFold(mem_copy_ops[mem_copy_ops.size() - 2],
                        mem_copy_ops.back())) {
    mem_copy_ops.pop_back();
  }

  int i = 0;
  for (auto &x : mem_copy_ops) {
    if (x.src_count > 1 || x.count > 1) {
      rocm_Broadcast_fp32(AsGpuStreamValue(stream),
                          reinterpret_cast<float *>(x.dst_ptr),
                          x.dst_stride >> 2, x.count, x.src_count,
                          reinterpret_cast<float *>(x.src_ptr), x.size >> 2);
    } else {
      DeviceMemoryBase src_mem = DeviceMemoryBase(x.src_ptr, x.size);
      DeviceMemoryBase target_mem = DeviceMemoryBase(x.dst_ptr, x.size);
      bool a_status = stream->ThenMemcpy(&target_mem, src_mem, x.size).ok();
      if (!a_status) {
        return tsl::Status(
            absl::StatusCode::kInternal,
            "failed to copy device memory in ROCMBlas::DoBlasGemmBatched");
      }
    }
    i++;
  }
  return tsl::OkStatus();
}

template <typename T>
tsl::Status ROCMBlas::AllocateStridedBuffer(
    const std::vector<typename RocBlasTypeConversionHelper<T>::mapped_type *>
        &raw_ptrs,
    int batch_count, uint64_t batch_stride, ScratchAllocator *scratch_allocator,
    Stream *stream,
    std::unique_ptr<TemporaryDeviceMemory<
        typename RocBlasTypeConversionHelper<T>::mapped_type>> *temp_memory,
    DeviceMemory<typename RocBlasTypeConversionHelper<T>::mapped_type>
        *device_memory,
    bool copy_data, bool &reallocated) {
  assert(device_memory != nullptr);

  using MAPPED_T = typename RocBlasTypeConversionHelper<T>::mapped_type;

  bool needs_allocate_strided = false;
  for (int i = 1; i < batch_count; ++i) {
    uint64_t tmp_batch_stride = raw_ptrs[i] - raw_ptrs[i - 1];
    if (tmp_batch_stride != batch_stride) {
      needs_allocate_strided = true;
      break;
    }
  }

  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);
  size_t matrix_batch_byte_size = matrix_byte_size * batch_count;

  // No need to do re-allocation, take the short cut and return
  if (!needs_allocate_strided) {
    *device_memory = DeviceMemory<MAPPED_T>(
        DeviceMemoryBase(raw_ptrs[0], matrix_batch_byte_size));
    reallocated = false;
    return tsl::OkStatus();
  }

  if (scratch_allocator != nullptr) {
    TF_ASSIGN_OR_RETURN(
        DeviceMemory<uint8> batch_matrix_bytes,
        scratch_allocator->AllocateBytes(matrix_batch_byte_size));
    *device_memory = DeviceMemory<MAPPED_T>(batch_matrix_bytes);
  } else {
    assert(temp_memory != nullptr);
    TF_ASSIGN_OR_RETURN(*temp_memory, stream->AllocateTemporaryArray<MAPPED_T>(
                                          matrix_batch_byte_size));
    *device_memory =
        DeviceMemory<MAPPED_T>(*(*temp_memory)->mutable_device_memory());
  }
  assert(batch_count > 0);
  char* device_memory_ptr = static_cast<char*>(device_memory->opaque());
  char* src_ptr = reinterpret_cast<char*>(raw_ptrs[0]);
  char* dst_ptr = device_memory_ptr;
  uint64_t cur_stride_size = matrix_byte_size;

  reallocated = true;

  if (copy_data)
    return ReorganizeMemory(stream, device_memory, raw_ptrs, batch_count,
                            batch_stride, true);
  return tsl::OkStatus();
}


template <typename T, typename FuncT>
tsl::Status ROCMBlas::DoBlasGemmBatchedInternal(
    FuncT rocblas_func, Stream *stream, const blas::BatchedGemmCall<T>& call) {
  using MAPPED_T = typename RocBlasTypeConversionHelper<T>::mapped_type;
  blas::Transpose transa = call.transa;
  blas::Transpose transb = call.transb;
  uint64_t m = call.m;
  uint64 n = call.n;
  uint64_t k = call.k;
  //blas::DataType dtype = ;
  DeviceMemorySlice<T> a_ptrs_to_wrappers = *call.pa;
  int lda = call.lda;
  DeviceMemorySlice<T> b_ptrs_to_wrappers = *call.pb;
  int ldb = call.ldb;
  DeviceMemorySlice<T> c_ptrs_to_wrappers = *call.pc;
  int ldc = call.ldc;
  blas::ComputePrecision precision = call.precision;
  ScratchAllocator* scratch_allocator = call.scratch_allocator;
  int batch_count = call.batch_count;

  if(!(int(call.context) & int(blas::CallContext::kSet)))
    throw tsl::errors::Internal("ROCMBlas::DoBlasGemmBatched did not receive gradient flags");

  // Sanity checks before making any further progress
  uint64_t batch_stride_a = 0;
  uint64_t batch_stride_b = 0;
  uint64_t batch_stride_c = 0;

  assert(ldc >= m);
  batch_stride_c = ldc * n;

  if (ROCMBlasTranspose(transa) == rocblas_operation_none) {
    assert(lda >= m);
    batch_stride_a = lda * k;
  } else {
    assert(lda >= k);
    batch_stride_a = lda * m;
  }

  if (ROCMBlasTranspose(transb) == rocblas_operation_none) {
    assert(ldb >= k);
    batch_stride_b = ldb * n;
  } else {
    assert(ldb >= n);
    batch_stride_b = ldb * k;
  }

  // Allocate local vectors to hold device pointers to matrices
  std::vector<MAPPED_T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    // static_cast does work when converting Eigen::half* to rocblas_half*,
    // hence the use of reinterpret_cast
    a_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  DeviceMemory<MAPPED_T> a;
  // Make sure the temporary memory are in-scope before the function returns
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> a_temp;
  bool reallocated_a, reallocated_b, reallocated_c;
  tsl::Status a_allocation_status = AllocateStridedBuffer<MAPPED_T>(
      a_raw_ptrs, batch_count, batch_stride_a, scratch_allocator, stream,
      &a_temp, &a, true, reallocated_a);
  if (a_allocation_status != tsl::OkStatus()) {
    return a_allocation_status;
  }

  DeviceMemory<MAPPED_T> b;
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> b_temp;
  tsl::Status b_allocation_status = AllocateStridedBuffer<MAPPED_T>(
      b_raw_ptrs, batch_count, batch_stride_b, scratch_allocator, stream,
      &b_temp, &b, true, reallocated_b);
  if (b_allocation_status != tsl::OkStatus()) {
    return b_allocation_status;
  }

  DeviceMemory<MAPPED_T> c;
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> c_temp;
  tsl::Status c_allocation_status = AllocateStridedBuffer<MAPPED_T>(
      c_raw_ptrs, batch_count, batch_stride_c, scratch_allocator, stream,
      &c_temp, &c, true, reallocated_c);  // can disable copy if beta=0
  if (c_allocation_status != tsl::OkStatus()) {
    return c_allocation_status;
  }

  bool ok;
  if constexpr (std::is_same_v<T, Eigen::bfloat16>) {
    float alpha = static_cast<float>(call.alpha);
    float beta = static_cast<float>(call.beta);
    ok = DoBlasInternal(
        rocblas_func, stream, /* pointer_mode_host = */ true,
        ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
        &alpha, a.opaque(), rocblas_datatype_bf16_r, lda, batch_stride_a,
        b.opaque(), rocblas_datatype_bf16_r, ldb, batch_stride_b, &beta,
        c.opaque(), rocblas_datatype_bf16_r, ldc, batch_stride_c, c.opaque(),
        rocblas_datatype_bf16_r, ldc, batch_stride_c, batch_count,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
  } else {
    T alpha = T(call.alpha), beta = T(call.beta);
    MAPPED_T *alpha_ptr = reinterpret_cast<MAPPED_T *>(&alpha);
    MAPPED_T *beta_ptr = reinterpret_cast<MAPPED_T *>(&beta);
    ok = DoBlasInternal(rocblas_func, stream, /* pointer_mode_host = */ true,
                        ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m,
                        n, k, alpha_ptr, GpuMemory(a), lda,
                        batch_stride_a, GpuMemory(b), ldb, batch_stride_b,
                        beta_ptr, GpuMemoryMutable(&c), ldc,
                        batch_stride_c, batch_count);
  }
  if (!ok)
    return tsl::Status(absl::StatusCode::kInternal,
                       "failed BLAS call, see log for details");
  if (reallocated_c)
    return ReorganizeMemory(stream, &c, c_raw_ptrs, batch_count, batch_stride_c,
                            false);
  return tsl::OkStatus();
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<Eigen::bfloat16> call) {
  return DoBlasGemmBatchedInternal(/*ignored*/wrap::rocblas_gemm_strided_batched_ex, stream, call).ok();
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<Eigen::half> call) {
  return DoBlasGemmBatchedInternal(wrap::rocblas_hgemm_strided_batched, stream, call).ok();
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<float> call) {
  return DoBlasGemmBatchedInternal(wrap::rocblas_sgemm_strided_batched, stream, call).ok();
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<double> call) {
  return DoBlasGemmBatchedInternal(wrap::rocblas_dgemm_strided_batched, stream, call).ok();
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<std::complex<float> > call) {
  return DoBlasGemmBatchedInternal(wrap::rocblas_cgemm_strided_batched, stream, call).ok();
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<std::complex<double> > call) {
  return DoBlasGemmBatchedInternal(wrap::rocblas_zgemm_strided_batched, stream, call).ok();
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  blas_log("DoBlasTrsm");
  return DoBlasInternal(wrap::rocblas_strsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(a), lda,
                        GpuMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  blas_log("DoBlasTrsm");
  return DoBlasInternal(wrap::rocblas_dtrsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(a), lda,
                        GpuMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return DoBlasInternal(wrap::rocblas_ctrsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return DoBlasInternal(wrap::rocblas_ztrsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb);
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 float alpha, const DeviceMemory<float *> &as,
                                 int lda, DeviceMemory<float *> *bs, int ldb,
                                 int batch_count) {
  return DoBlasInternal(wrap::rocblas_strsm_batched, stream,
                        true /* = pointer_mode_host */, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(as),
                        lda, GpuMemoryMutable(bs), ldb, batch_count);
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 double alpha, const DeviceMemory<double *> &as,
                                 int lda, DeviceMemory<double *> *bs, int ldb,
                                 int batch_count) {
  return DoBlasInternal(wrap::rocblas_dtrsm_batched, stream,
                        true /* = pointer_mode_host */, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(as),
                        lda, GpuMemoryMutable(bs), ldb, batch_count);
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<float> alpha,
                                 const DeviceMemory<std::complex<float> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<float> *> *bs,
                                 int ldb, int batch_count) {
  return DoBlasInternal(
      wrap::rocblas_ctrsm_batched, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
      static_cast<const rocblas_float_complex *const *>(as.opaque()), lda,
      static_cast<rocblas_float_complex *const *>(bs->opaque()), ldb,
      batch_count);
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<double> alpha,
                                 const DeviceMemory<std::complex<double> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<double> *> *bs,
                                 int ldb, int batch_count) {
  return DoBlasInternal(
      wrap::rocblas_ztrsm_batched, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
      static_cast<const rocblas_double_complex *const *>(as.opaque()), lda,
      static_cast<rocblas_double_complex *const *>(bs->opaque()), ldb,
      batch_count);
}

template <typename T, typename U, typename V>
tsl::Status ROCMBlas::DoBlasGemmStridedInternalNonEx(Stream *stream, V fun,
    const blas::GemmCall& call) {
  blas::Transpose transa = call.transa;
  blas::Transpose transb = call.transb;
  uint64_t m = call.m;
  uint64 n = call.n;
  uint64_t k = call.k;
  blas::DataType dtype = call.dtype_in;
  const DeviceMemoryBase &a = *call.pa;
  int lda = call.lda;
  const DeviceMemoryBase &b = *call.pb;
  int ldb = call.ldb;
  DeviceMemoryBase *c = call.c;
  int ldc = call.ldc;
  blas::ComputePrecision precision = call.precision;
  int stride_a = call.stride_a;
  int stride_b = call.stride_b;
  int stride_c = call.stride_c;
  int batch_count = call.batch_count;

  auto alpha = alpha_cast<T, U>(call.alpha, call.dtype_ab, 1);
  auto beta = alpha_cast<T, U>(call.beta, call.dtype_ab, 0);
  bool pointer_mode_host = std::is_same_v<U, std::complex<float> >
    || std::is_same_v<U, std::complex<double> >;
  return DoBlasInternalStatus(fun, stream,
          pointer_mode_host,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          &alpha,
          reinterpret_cast<const U *>(a.opaque()), lda, stride_a,
          reinterpret_cast<const U *>(b.opaque()), ldb, stride_b,
          &beta,
          reinterpret_cast<U *>(c->opaque()), ldc, stride_c,
          batch_count);
}

tsl::Status ROCMBlas::DoBlasGemmStridedInternalEx(Stream *stream,
    const blas::GemmCall& call, rocblas_datatype dt) {
  blas::Transpose transa = call.transa;
  blas::Transpose transb = call.transb;
  uint64_t m = call.m;
  uint64 n = call.n;
  uint64_t k = call.k;
  blas::DataType dtype = call.dtype_in;
  const DeviceMemoryBase &a = *call.pa;
  int lda = call.lda;
  const DeviceMemoryBase &b = *call.pb;
  int ldb = call.ldb;
  DeviceMemoryBase *c = call.c;
  int ldc = call.ldc;
  blas::ComputePrecision precision = call.precision;
  int stride_a = call.stride_a;
  int stride_b = call.stride_b;
  int stride_c = call.stride_c;
  int batch_count = call.batch_count;

  auto alpha = alpha_cast<float>(call.alpha, call.dtype_ab, 1);
  auto beta = alpha_cast<float>(call.beta, call.dtype_ab, 0);
  return DoBlasInternalStatus(
      wrap::rocblas_gemm_strided_batched_ex, stream,
      false, /* pointer_mode_host */
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, 
      &alpha,
      a.opaque(), dt, lda, stride_a, b.opaque(),
      dt, ldb, stride_b, 
      &beta, 
      c->opaque(),
      dt, ldc, stride_c, c->opaque(),
      dt, ldc, stride_c, batch_count,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
}

tsl::Status ROCMBlas::DoBlasGemmInternalEx2(Stream* stream,
  const blas::GemmCall& call) {
#if ROCBLAS_VERSION_MAJOR>3 || (ROCBLAS_VERSION_MAJOR==3 && ROCBLAS_VERSION_MINOR>=1) 
  blas::Transpose transa = call.transa;
  blas::Transpose transb = call.transb;
  uint64_t m = call.m;
  uint64 n = call.n;
  uint64_t k = call.k;
  blas::DataType dtype = call.dtype_in;
  const DeviceMemoryBase &a = *call.pa;
  int lda = call.lda;
  const DeviceMemoryBase &b = *call.pb;
  int ldb = call.ldb;
  DeviceMemoryBase *c = call.c;
  int ldc = call.ldc;
  blas::ComputePrecision precision = call.precision;
/*
  int stride_a = call.stride_a;
  int stride_b = call.stride_b;
  int stride_c = call.stride_c;
  int batch_count = call.batch_count;
 */
  auto alpha = alpha_cast<float>(call.alpha, call.dtype_ab, 1);
  auto beta = alpha_cast<float>(call.beta, call.dtype_ab, 0);

  rocblas_datatype compute_type;
  if(!(call.context & blas::CallContext::kEnableF8)) {
      compute_type = (rocblas_datatype)rocblas_compute_type_f32;
  } else {
    switch (int(call.context) & 3) {
            case 0:
              compute_type = (rocblas_datatype)rocblas_compute_type_f8_f8_f32;
              break;
            case 1:
              compute_type = (rocblas_datatype)rocblas_compute_type_bf8_f8_f32;
              break;
            case 2:          
              compute_type = (rocblas_datatype)rocblas_compute_type_f8_bf8_f32;
              break;
            case 3:
              return tsl::errors::Internal(absl::StrCat("Unexpected grad_flags for GEMM: ",
                                                int(call.context)));
    }
  }
  rocblas_stride stride_ar = lda, stride_ac = 1;
  rocblas_stride stride_br = ldb, stride_bc = 1;
  rocblas_stride stride_cr = ldc, stride_cc = 1;
  if(transa == blas::Transpose::kTranspose)
    std::swap(stride_ar, stride_ac);
  if(transb == blas::Transpose::kTranspose)
    std::swap(stride_br, stride_bc);
  return DoBlasInternalStatus(
          wrap::rocblas_gemm_ext2, stream, /*ignored*/ true,
          (rocblas_int)m, (rocblas_int)n, (rocblas_int)k, 
          &alpha, a.opaque(), rocblas_datatype_f16_r, stride_ar, stride_ac,
          b.opaque(), rocblas_datatype_f16_r, stride_br, stride_bc,
          &beta, 
          c->opaque(), rocblas_datatype_f16_r, stride_cr, stride_cc,
          c->opaque(), rocblas_datatype_f16_r, stride_cr, stride_cc,
          compute_type,
          rocblas_gemm_algo_standard, 0, 0x18);
#else
  return tsl::errors::Internal("Not implemented");
#endif 
}


tsl::Status ROCMBlas::DoBlasGemmStridedInternalEx2(Stream* stream,
  const blas::GemmCall& call) {
  return DoBlasGemmInternalEx2(stream, call); 
#if 0  
#if ROCBLAS_VERSION_MAJOR>3 || (ROCBLAS_VERSION_MAJOR==3 && ROCBLAS_VERSION_MINOR>=1) 
  blas::Transpose transa = call.transa;
  blas::Transpose transb = call.transb;
  uint64_t m = call.m;
  uint64 n = call.n;
  uint64_t k = call.k;
  blas::DataType dtype = call.dtype_in;
  const DeviceMemoryBase &a = *call.pa;
  int lda = call.lda;
  const DeviceMemoryBase &b = *call.pb;
  int ldb = call.ldb;
  DeviceMemoryBase *c = call.c;
  int ldc = call.ldc;
  blas::ComputePrecision precision = call.precision;
  int stride_a = call.stride_a;
  int stride_b = call.stride_b;
  int stride_c = call.stride_c;
  int batch_count = call.batch_count;
  auto alpha = alpha_cast<float>(call.alpha, call.dtype_ab, 1);
  auto beta = alpha_cast<float>(call.beta, call.dtype_ab, 0);
 
  rocblas_computetype compute_type;
  switch (int(call.context) & 3) {
          case 0:
            compute_type = rocblas_compute_type_f8_f8_f32;
            break;
          case 1:
            compute_type = rocblas_compute_type_bf8_f8_f32;
            break;
          case 2:          
            compute_type = rocblas_compute_type_f8_bf8_f32;
            break;
          case 3:
            return tsl::errors::Internal(absl::StrCat("Unexpected grad_flags for GEMM: ",
                                              int(call.context)));
    }
    rocblas_stride stride_ar = lda, stride_ac = 1;
    rocblas_stride stride_br = ldb, stride_bc = 1;
    rocblas_stride stride_cr = ldc, stride_cc = 1;
    if(transa == blas::Transpose::kTranspose)
      std::swap(stride_ar, stride_ac);
    if(transb == blas::Transpose::kTranspose)
      std::swap(stride_br, stride_bc);
    return DoBlasInternalStatus(
            wrap::rocblas_gemm_ext2, stream, /*ignored*/ true,
            (rocblas_int)m, (rocblas_int)n, (rocblas_int)k, 
            &alpha, a.opaque(), rocblas_datatype_f16_r, stride_ar, stride_ac,
            b.opaque(), rocblas_datatype_f16_r, stride_br, stride_bc,
            &beta, 
            c->opaque(), rocblas_datatype_f16_r, stride_cr, stride_cc,
            c->opaque(), rocblas_datatype_f16_r, stride_cr, stride_cc,
            compute_type,
            rocblas_gemm_algo_standard, 0, 0x18);
#else
  return tsl::errors::Internal("Not implemented");
#endif
#endif
}

tsl::Status ROCMBlas::DoBlasGemmStridedBatched(Stream *stream, const blas::GemmCall& call) {
  switch (call.dtype_in) {
    case blas::DataType::kHalf:
      return DoBlasGemmStridedInternalNonEx<Eigen::half, rocblas_half>(stream, 
        wrap::rocblas_hgemm_strided_batched, call);
    case blas::DataType::kFloat:
      return DoBlasGemmStridedInternalNonEx<float, float>(stream, 
        wrap::rocblas_sgemm_strided_batched, call);
    case blas::DataType::kDouble:
      return DoBlasGemmStridedInternalNonEx<double, double>(stream, 
        wrap::rocblas_dgemm_strided_batched, call);
    case blas::DataType::kComplexFloat:
      return DoBlasGemmStridedInternalNonEx<std::complex<float>, rocblas_float_complex>(stream, 
        wrap::rocblas_cgemm_strided_batched, call);
    case blas::DataType::kComplexDouble:
      return DoBlasGemmStridedInternalNonEx<std::complex<double>, rocblas_double_complex>(stream, 
        wrap::rocblas_zgemm_strided_batched, call);
    case blas::DataType::kBF16:
      return DoBlasGemmStridedInternalEx(stream, call, rocblas_datatype_bf16_r);
    default:
      return tsl::errors::Internal(absl::StrCat(
          "Unsupported datatype for GEMM: ", blas::DataTypeString(call.dtype_in)));
  }
}

tsl::Status ROCMBlas::GetVersion(string *version) {
  return tsl::errors::Unimplemented("");
}

}  // namespace gpu

void initialize_rocblas() {
  auto rocBlasAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      rocm::kROCmPlatformId, PluginKind::kBlas, gpu::kRocBlasPlugin);

  if (!rocBlasAlreadyRegistered) {
    tsl::Status status =
        PluginRegistry::Instance()
            ->RegisterFactory<PluginRegistry::BlasFactory>(
                rocm::kROCmPlatformId, gpu::kRocBlasPlugin, "rocBLAS",
                [](internal::StreamExecutorInterface *parent)
                    -> blas::BlasSupport * {
                  gpu::GpuExecutor *rocm_executor =
                      dynamic_cast<gpu::GpuExecutor *>(parent);
                  if (rocm_executor == nullptr) {
                    LOG(ERROR)
                        << "Attempting to initialize an instance of the "
                           "rocBLAS "
                        << "support library with a non-ROCM StreamExecutor";
                    return nullptr;
                  }

                  gpu::ROCMBlas *blas = new gpu::ROCMBlas(rocm_executor);
                  if (!blas->Init()) {
                    // Note: Init() will log a more specific error.
                    delete blas;
                    return nullptr;
                  }
                  return blas;
                });

    if (!status.ok()) {
      LOG(ERROR) << "Unable to register rocBLAS factory: " << status.message();
    }

    PluginRegistry::Instance()->SetDefaultFactory(
        rocm::kROCmPlatformId, PluginKind::kBlas, gpu::kRocBlasPlugin);
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_rocblas,
                            { stream_executor::initialize_rocblas(); });
