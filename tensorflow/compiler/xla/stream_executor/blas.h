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

// Exposes the family of BLAS routines as pre-canned high performance calls for
// use in conjunction with the StreamExecutor abstraction.
//
// Note that this interface is optionally supported by platforms; see
// StreamExecutor::SupportsBlas() for details.
//
// This abstraction makes it simple to entrain BLAS operations on GPU data into
// a Stream -- users typically will not use this API directly, but will use the
// Stream builder methods to entrain these operations "under the hood". For
// example:
//
//  DeviceMemory<float> x = stream_exec->AllocateArray<float>(1024);
//  DeviceMemory<float> y = stream_exec->AllocateArray<float>(1024);
//  // ... populate x and y ...
//  Stream stream{stream_exec};
//  stream
//    .Init()
//    .ThenBlasAxpy(1024, 5.5, x, 1, &y, 1);
//  TF_CHECK_OK(stream.BlockHostUntilDone());
//
// By using stream operations in this manner the user can easily intermix custom
// kernel launches (via StreamExecutor::ThenLaunch()) with these pre-canned BLAS
// routines.

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_BLAS_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_BLAS_H_

#include <complex>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/stream_executor/data_type.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/protobuf/dnn.pb.h"

namespace Eigen {
struct half;
}  // namespace Eigen

namespace stream_executor {

class Stream;
class ScratchAllocator;

template <typename ElemT>
class DeviceMemory;

template <typename ElemT>
class HostOrDeviceScalar;

template <typename T>
using DeviceMemorySlice = absl::Span<DeviceMemory<T> *const>;

namespace blas {

// Specifies whether the input matrix will be transposed or
// transposed+conjugated before any BLAS operations.
enum class Transpose { kNoTranspose, kTranspose, kConjugateTranspose };

// Returns a name for t.
std::string TransposeString(Transpose t);

// Specifies whether the upper or lower triangular part of a
// symmetric/Hermitian matrix is used.
enum class UpperLower { kUpper, kLower };

// Returns a name for ul.
std::string UpperLowerString(UpperLower ul);

// Specifies whether a matrix is unit triangular.
enum class Diagonal { kUnit, kNonUnit };

// Returns a name for d.
std::string DiagonalString(Diagonal d);

// Specifies whether a Hermitian matrix appears on the left or right in
// operation.
enum class Side { kLeft, kRight };

// Returns a name for s.
std::string SideString(Side s);

// Type with which intermediate computations of a blas routine are performed.
//
// Some blas calls can perform computations with a type that's different than
// the type of their inputs/outputs.  This lets you e.g. multiply two matrices
// of int8s using float32s to store the matmul's intermediate values.
enum class ComputationType {
  kF16,  // 16-bit floating-point
  kF32,  // 32-bit floating-point
  kF64,  // 64-bit floating-point
  kI32,  // 32-bit integer
  // The below values use float32 for accumulation, but allow the inputs and
  // outputs to be downcast to a lower precision:
  kF16AsF32,   // Allow downcast to F16 precision.
  kBF16AsF32,  // Allow downcast to BF16 precision.
  kTF32AsF32,  // Allow downcast to TF32 precision.

  kUndefined,
};

// Converts a ComputationType to a string.
std::string ComputationTypeString(ComputationType ty);

std::ostream &operator<<(std::ostream &os, ComputationType ty);

using dnn::DataType;
using dnn::ToDataType;

// Converts a ComputationType to a string.
std::string DataTypeString(DataType ty);

std::ostream &operator<<(std::ostream &os, DataType ty);

// Opaque identifier for an "algorithm" used by a blas routine.  This functions
// as a hint to the blas library.
typedef int64_t AlgorithmType;
constexpr AlgorithmType kDefaultAlgorithm = -1;
constexpr AlgorithmType kDefaultBlasGemm = -2;
constexpr AlgorithmType kDefaultBlasGemv = -3;
constexpr AlgorithmType kNoAlgorithm = -4;
constexpr AlgorithmType kRuntimeAutotuning = -5;

// blas uses -1 to represent the default algorithm. This happens to match up
// with the CUBLAS_GEMM_DFALT constant, so cuda_blas.cc is using static_cast
// to convert from AlgorithmType to cublasGemmAlgo_t, and uses a static_assert
// to ensure that this assumption does not break.
// If another blas implementation uses a different value for the default
// algorithm, then it needs to convert kDefaultGemmAlgo to that value
// (e.g. via a function called ToWhateverGemmAlgo).
constexpr AlgorithmType kDefaultGemmAlgo = -1;

// Describes the result of a performance experiment, usually timing the speed of
// a particular AlgorithmType.
//
// If the call we were benchmarking failed (a common occurrence; not all
// algorithms are valid for all calls), is_valid() will be false.
class ProfileResult {
 public:
  bool is_valid() const { return is_valid_; }
  void set_is_valid(bool val) { is_valid_ = val; }
  AlgorithmType algorithm() const { return algorithm_; }
  void set_algorithm(AlgorithmType val) { algorithm_ = val; }
  float elapsed_time_in_ms() const { return elapsed_time_in_ms_; }
  void set_elapsed_time_in_ms(float val) { elapsed_time_in_ms_ = val; }

 private:
  bool is_valid_ = false;
  AlgorithmType algorithm_ = kDefaultAlgorithm;
  float elapsed_time_in_ms_ = std::numeric_limits<float>::max();
};

class AlgorithmConfig {
 public:
  AlgorithmConfig() : algorithm_(kDefaultAlgorithm) {}
  explicit AlgorithmConfig(AlgorithmType algorithm) : algorithm_(algorithm) {}
  AlgorithmType algorithm() const { return algorithm_; }
  void set_algorithm(AlgorithmType val) { algorithm_ = val; }
  bool operator==(const AlgorithmConfig &other) const {
    return this->algorithm_ == other.algorithm_;
  }
  bool operator!=(const AlgorithmConfig &other) const {
    return !(*this == other);
  }
  std::string ToString() const;

 private:
  AlgorithmType algorithm_;
};

// Opaque identifier specifying the precision to use in gemm calls.
typedef int64_t ComputePrecision;
constexpr ComputePrecision kDefaultComputePrecision = 0;

// Call context information for GEMM API calls
// This is extra information that can optionally be passed down to the blas
// library, so that it can pick the efficient imlpementation based on context
enum class CallContext {
  kNone = 0,            // No information
  kGradient1 = 1,
  kGradient2 = 2,
  kEnableF8 = 4,
  kSet = 256
};

inline CallContext operator|(CallContext a, CallContext b) {
  return static_cast<CallContext>(static_cast<int>(a) | static_cast<int>(b));
}

inline bool operator&(CallContext a, CallContext b) {
  return static_cast<int>(a) & static_cast<int>(b);
}

template <class T>
struct GemmCallAlpha
{
  typedef T type;
};

template <>
struct GemmCallAlpha<Eigen::half>
{
  typedef float type;
};

template <>
struct GemmCallAlpha<Eigen::bfloat16>
{
  typedef float type;
};

struct GemmCall {
  blas::Transpose transa, transb;
  uint64_t m; 
  uint64_t n;
  uint64_t k;
  const void *alpha = nullptr;
  const DeviceMemoryBase* pa; 
  int lda;
  const DeviceMemoryBase* pb;
  int ldb;
  const void *beta = nullptr;
  DeviceMemoryBase *c;
  int ldc;

  DataType dtype_in, dtype_out, dtype_ab;

  int64_t stride_a = -1;
  int64_t stride_b = -1;
  int64_t stride_c = -1;
  int batch_count = 1;
  CallContext context = CallContext::kNone;
  ComputationType type = ComputationType::kUndefined;
  ComputePrecision precision = kDefaultComputePrecision;
  ProfileResult *output_profile_result = 0;
  AlgorithmType algorithm = kDefaultGemmAlgo;

  GemmCall() {}
  template <typename T, typename U, typename V=T>
  GemmCall(blas::Transpose _transa,
          blas::Transpose _transb,
          uint64_t _m,
          uint64_t _n,
          uint64_t _k,
          const DeviceMemory<T> &_a, int _lda,
          const DeviceMemory<T> &_b, int _ldb,
          DeviceMemory<U> *_c, int _ldc,
          const V* _alpha = nullptr,
          const V* _beta = nullptr
          )
  : transa(_transa), transb(_transb), m(_m), n(_n), k(_k),
    alpha(_alpha), 
    pa(&_a), lda(_lda), pb(&_b), ldb(_ldb), 
    beta(_beta),
    c(_c), ldc(_ldc)
  {
    dtype_in = blas::ToDataType<T>::value;
    dtype_out = blas::ToDataType<U>::value;
    dtype_ab = blas::ToDataType<V>::value;
  }
};


template <class T>
struct BatchedGemmCall
{
  using DeviceMemorySlice = absl::Span<DeviceMemory<T> *const>;

  blas::Transpose transa;
  blas::Transpose transb;
  uint64 m, n, k;
  typename GemmCallAlpha<T>::type alpha;
  DeviceMemorySlice *pa;
  int lda;
  DeviceMemorySlice *pb;
  int ldb;
  typename GemmCallAlpha<T>::type beta;
  DeviceMemorySlice *pc;
  int ldc;
  int batch_count;
  CallContext context = CallContext::kNone;
  ScratchAllocator *scratch_allocator = nullptr;
  ComputationType type = ComputationType::kUndefined;
  ComputePrecision precision = kDefaultComputePrecision;
  ProfileResult *output_profile_result = 0;
  AlgorithmType algorithm = kDefaultGemmAlgo;
};

// BLAS support interface -- this can be derived from a GPU executor when the
// underlying platform has an BLAS library implementation available. See
// StreamExecutor::AsBlas().
//
// Thread-hostile: CUDA associates a CUDA-context with a particular thread in
// the system. Any operation that a user attempts to perform by enqueueing BLAS
// operations on a thread not-associated with the CUDA-context has unknown
// behavior at the current time; see b/13176597
class BlasSupport {
 public:
  virtual ~BlasSupport() {}

  // Performs a BLAS y <- ax+y operation.
  virtual bool DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasAxpy(Stream *stream, uint64_t elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) = 0;
  virtual bool DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Copies vector to another vector: y <- x.
  virtual bool DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) = 0;
  virtual bool DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Computes the product of a vector by a scalar: x <- a*x.
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) = 0;

  // Computes a matrix-vector product using a general matrix.
  //
  //     y <- alpha * a * x + beta * y,
  // or
  //     y <- alpha * a' * x + beta * y,
  // or
  //     y <- alpha * conj(a') * x + beta * y,
  //
  // alpha and beta are scalars; a is an m-by-n general matrix; x is a vector
  // with n(trans==kNoTranspose)/m(otherwise) elements;
  // y is a vector with m(trans==kNoTranspose)/n(otherwise) elements.
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) = 0;
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Computes a matrix-vector product using a symmetric band matrix.
  //
  //     y <- alpha * a * x + beta * y,
  //
  // alpha and beta are scalars; a is an n-by-n symmetric band matrix, with k
  // super-diagonals; x and y are n-element vectors.
  virtual bool DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) = 0;

  // Computes a matrix-matrix product with general matrices:
  //
  //     c <- alpha * op(a) * op(b) + beta * c,
  //
  // op(X) is one of op(X) = X, or op(X) = X', or op(X) = conj(X'); alpha and
  // beta are scalars; a, b, and c are matrices; op(a) is an m-by-k matrix;
  // op(b) is a k-by-n matrix; c is an m-by-n matrix.
  //
  // Note: The half interface uses float precision internally; the version
  // that uses half precision internally is not yet supported. There is no
  // batched version of the half-precision interface.
  //
  // Alpha/beta type matches `dtype`, unless `dtype` is `Eigen::half`, in that
  // case the expected alpha/beta type is `float`.

  // covers: DoBlasGemm, DoBlasGemmWithAlgorithm, 
  // DoBlasGemmStridedBatched, DoBlasGemmStridedBatchedWithAlgorithm
  virtual tsl::Status DoBlasGemm(Stream *stream, const GemmCall& call) = 0;

  // Gets a list of supported algorithms for DoBlasGemmWithAlgorithm.
  virtual bool GetBlasGemmAlgorithms(
      Stream *stream, std::vector<AlgorithmType> *out_algorithms) = 0;

  // Computes a batch of matrix-matrix product with general matrices.
  // This is a batched version of DoBlasGemm.
  // The batched GEMM computes matrix product for each input/output in a, b,
  // and c, which contain batch_count DeviceMemory objects.
  virtual bool DoBlasGemmBatched(Stream *stream, BatchedGemmCall<Eigen::half> c) = 0;
  virtual bool DoBlasGemmBatched(Stream *stream, BatchedGemmCall<Eigen::bfloat16> c) = 0;
  virtual bool DoBlasGemmBatched(Stream *stream, BatchedGemmCall<float> c) = 0;
  virtual bool DoBlasGemmBatched(Stream *stream, BatchedGemmCall<double> c) = 0;
  virtual bool DoBlasGemmBatched(Stream *stream, BatchedGemmCall<std::complex<float> > c) = 0;
  virtual bool DoBlasGemmBatched(Stream *stream, BatchedGemmCall<std::complex<double> > c) = 0;

  // Solves a triangular matrix equation.
  //
  //     op(a) * x = alpha * b,
  // or
  //     x * op(a) = alpha * b
  //
  // alpha is a scalar; x and b are m-by-n matrices; a is a unit, or non-unit,
  // upper or lower triangular matrix; op(a) is one of op(a) = a, or op(a) = a',
  // or op(a) = conj(a').
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) = 0;
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) = 0;
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) = 0;
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) = 0;

  // Same as DoBlasTrsm, but operates over a list of a's and b's.  The lists
  // `as` and `bs` must have the same length.
  virtual bool DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 float alpha, const DeviceMemory<float *> &as,
                                 int lda, DeviceMemory<float *> *bs, int ldb,
                                 int batch_count) = 0;
  virtual bool DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 double alpha, const DeviceMemory<double *> &as,
                                 int lda, DeviceMemory<double *> *bs, int ldb,
                                 int batch_count) = 0;
  virtual bool DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<float> alpha,
                                 const DeviceMemory<std::complex<float> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<float> *> *bs,
                                 int ldb, int batch_count) = 0;
  virtual bool DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<double> alpha,
                                 const DeviceMemory<std::complex<double> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<double> *> *bs,
                                 int ldb, int batch_count) = 0;

  virtual tsl::Status GetVersion(std::string *version) = 0;

 protected:
  BlasSupport() {}

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(BlasSupport);
};

// Macro used to quickly declare overrides for abstract virtuals in the
// BlasSupport base class.
#define TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES                  \
  bool DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,            \
                  const DeviceMemory<float> &x, int incx,                      \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasAxpy(Stream *stream, uint64_t elem_count, double alpha,           \
                  const DeviceMemory<double> &x, int incx,                     \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasAxpy(Stream *stream, uint64_t elem_count,                         \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasAxpy(Stream *stream, uint64_t elem_count,                         \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasCopy(Stream *stream, uint64_t elem_count,                         \
                  const DeviceMemory<float> &x, int incx,                      \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasCopy(Stream *stream, uint64_t elem_count,                         \
                  const DeviceMemory<double> &x, int incx,                     \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasCopy(Stream *stream, uint64_t elem_count,                         \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasCopy(Stream *stream, uint64_t elem_count,                         \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,            \
                  DeviceMemory<float> *x, int incx) override;                  \
  bool DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,           \
                  DeviceMemory<double> *x, int incx) override;                 \
  bool DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,            \
                  DeviceMemory<std::complex<float>> *x, int incx) override;    \
  bool DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,           \
                  DeviceMemory<std::complex<double>> *x, int incx) override;   \
  bool DoBlasScal(Stream *stream, uint64_t elem_count,                         \
                  std::complex<float> alpha,                                   \
                  DeviceMemory<std::complex<float>> *x, int incx) override;    \
  bool DoBlasScal(Stream *stream, uint64_t elem_count,                         \
                  std::complex<double> alpha,                                  \
                  DeviceMemory<std::complex<double>> *x, int incx) override;   \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, \
                  float alpha, const DeviceMemory<float> &a, int lda,          \
                  const DeviceMemory<float> &x, int incx, float beta,          \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, \
                  double alpha, const DeviceMemory<double> &a, int lda,        \
                  const DeviceMemory<double> &x, int incx, double beta,        \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n, uint64 k, \
                  float alpha, const DeviceMemory<float> &a, int lda,          \
                  const DeviceMemory<float> &x, int incx, float beta,          \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n, uint64 k, \
                  double alpha, const DeviceMemory<double> &a, int lda,        \
                  const DeviceMemory<double> &x, int incx, double beta,        \
                  DeviceMemory<double> *y, int incy) override;                 \
  tsl::Status DoBlasGemm(Stream *stream, const blas::GemmCall& call);          \
  bool GetBlasGemmAlgorithms(Stream *stream,                                   \
                             std::vector<blas::AlgorithmType> *out_algorithms) \
      override;                                                                \
  bool DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<Eigen::half> c)       \
      override;                                                                \
  bool DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<Eigen::bfloat16> c)   \
      override;                                                                \
  bool DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<float> c)             \
      override;                                                                \
  bool DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<double> c)            \
      override;                                                                \
  bool DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<std::complex<float>>) \
      override;                                                                \
  bool DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCall<std::complex<double>>)\
      override;                                                                \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,     \
                  uint64_t n, float alpha, const DeviceMemory<float> &a,       \
                  int lda, DeviceMemory<float> *b, int ldb) override;          \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,     \
                  uint64_t n, double alpha, const DeviceMemory<double> &a,     \
                  int lda, DeviceMemory<double> *b, int ldb) override;         \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,     \
                  uint64_t n, std::complex<float> alpha,                       \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  DeviceMemory<std::complex<float>> *b, int ldb) override;     \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,     \
                  uint64_t n, std::complex<double> alpha,                      \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  DeviceMemory<std::complex<double>> *b, int ldb) override;    \
  bool DoBlasTrsmBatched(                                                      \
      Stream *stream, blas::Side side, blas::UpperLower uplo,                  \
      blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64 n,       \
      float alpha, const DeviceMemory<float *> &as, int lda,                   \
      DeviceMemory<float *> *bs, int ldb, int batch_count) override;           \
  bool DoBlasTrsmBatched(                                                      \
      Stream *stream, blas::Side side, blas::UpperLower uplo,                  \
      blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64 n,       \
      double alpha, const DeviceMemory<double *> &as, int lda,                 \
      DeviceMemory<double *> *bs, int ldb, int batch_count) override;          \
  bool DoBlasTrsmBatched(Stream *stream, blas::Side side,                      \
                         blas::UpperLower uplo, blas::Transpose transa,        \
                         blas::Diagonal diag, uint64_t m, uint64 n,            \
                         std::complex<float> alpha,                            \
                         const DeviceMemory<std::complex<float> *> &as,        \
                         int lda, DeviceMemory<std::complex<float> *> *bs,     \
                         int ldb, int batch_count) override;                   \
  bool DoBlasTrsmBatched(Stream *stream, blas::Side side,                      \
                         blas::UpperLower uplo, blas::Transpose transa,        \
                         blas::Diagonal diag, uint64_t m, uint64 n,            \
                         std::complex<double> alpha,                           \
                         const DeviceMemory<std::complex<double> *> &as,       \
                         int lda, DeviceMemory<std::complex<double> *> *bs,    \
                         int ldb, int batch_count) override;                   \
  tsl::Status GetVersion(std::string *version) override;

}  // namespace blas
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_BLAS_H_
