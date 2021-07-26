/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XRT_XRT_COMPILATION_CACHE_H_
#define TENSORFLOW_COMPILER_XRT_XRT_COMPILATION_CACHE_H_

#include <memory>
#include <string>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xrt/xrt_refptr.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

extern const char* kXRTCompilationCacheResourceName;

struct XRTCompilationCacheEntry {
  explicit XRTCompilationCacheEntry(xla::LocalExecutable* executable)
      : executable(executable) {}

  // Returns a non-owned pointer to an immutable executable.
  xla::LocalExecutable* get_executable() const { return executable; }

 private:
  xla::LocalExecutable* executable;
};

// Base class for a reference to a cached executable. A unique_ptr to a
// XRTCompilationCacheEntryRef is returned by the cache Lookup methods below,
// and ensures the underlying executable is not garbage-collected until the
// client discards the ptr.
class XRTCompilationCacheEntryRef {
 public:
  virtual ~XRTCompilationCacheEntryRef() = default;

  // Returns a XRTCompilationCacheEntry that should not be used beyond the
  // lifetime of the XRTCompilationCacheEntryRef.
  virtual XRTCompilationCacheEntry get() = 0;
};

// Cache for compiled XLA executables.
// TODO(b/112646171) rationalize this with the other compilation caches.
//
// Each key identifies a unique XLA computation, and the value is executable
// generated by compiling the computation.
//
// When a computation is considered for compilation, the client calls
//
// auto key = <compute key for computation>;
// auto compile_function = <lambda to compile computation into executable>;
// int64 uid;
// CompileIfKeyAbsent(computation_key, &uid, compile_function);
//
// where computation_key is the key computed for the computation. On success,
// uid contains an identifier that can be used to look up the executable. If the
// compiled executable were not present in the cache, compile_function would be
// called to generate it.
//
// The caller is responsible for calling Release(uid) once for every
// call to CompileIfKeyAbsent(key, ...) to discard the reference to the
// compilation results, after the caller is sure it will not look up the
// compiled executables again.
//
// Subsequently the client can call
//
// std::unique_ptr<XRTCompilationCacheEntryRef> entry;
// Lookup(uid, &entry);
// auto proto = entry->get();
//
// to access a cached executable.
class XRTCompilationCache : public ResourceBase {
 public:
  // There is no way in general to discover the size taken by an XLA executable,
  // so the cache defaults to a specific number of entries to determine when to
  // start evicting programs. TODO(b/112592410) change this if the XLA API gets
  // a mechanism to query size.
  explicit XRTCompilationCache(int max_number_of_entries);
  ~XRTCompilationCache() override;

  // Ensures there is an entry for key present in the cache. By the time
  // CompileIfKeyAbsent returns there is guaranteed to be an entry in the cache
  // for key, and that entry will remain valid at least until Release is called
  // on the returned uid. The first call to CompileIfKeyAbsent with a key that
  // is not in the cache will evaluate compile_function to compute the value to
  // use in the entry. Subsequent calls with the same key will block until
  // compile_function completes. Other cache reads and inserts may proceed on
  // other threads while compile_function is executing. The caller is
  // responsible for calling Release(uid) to manually discard its reference to
  // the compiled program, once the caller will not look up the compiled program
  // again.
  //
  // compile_function should compile the computation represented by key and fill
  // the xla::LocalExecutable into its passed argument. It should return OK
  // if and only if compilation succeeds. The executable will be discarded on
  // non-OK status.
  Status CompileIfKeyAbsent(
      const string& key, int64* uid,
      const std::function<Status(std::unique_ptr<xla::LocalExecutable>*)>&
          compile_function);

  Status Release(int64_t uid);

  // Looks up an executable corresponding to uid. On success a pointer to an
  // EntryRef holding the program is returned in entry.
  Status Lookup(int64_t uid,
                std::unique_ptr<XRTCompilationCacheEntryRef>* entry);

  string DebugString() const override;

 private:
  // An entry in the compilation cache. The entry is deleted once it has been
  // marked for eviction from the cache _and_ all looked-up entries have been
  // released. When the entry is first created, it is uninitialized and a
  // client-supplied compilation function is run outside the cache's lock to
  // generate the program to be stored in the entry. Any other client that
  // requests the entry will block until it has been initialized. Each entry has
  // a last_use value that set from a monotonically-increasing counter in the
  // cache whenever the entry is referenced. When the cache becomes full,
  // entries are marked for eviction in LRU order.
  struct CompiledSubgraph : public core::RefCounted {
    ~CompiledSubgraph() override = default;

    XRTCompilationCache* parent = nullptr;  // Not owned.
    bool initialized = false;
    // The Status returned by the compilation function when the entry is
    // initialized. This status will be returned to any client that requests the
    // entry.
    Status initialization_status;
    // Counter to keep track of LRU entries for the eviction policy.
    int64 last_use = -1;
    // The unique key describing this entry.
    string key;
    // The uid describing this entry.
    int64 uid;
    // The compiled payload corresponding to the key.
    std::unique_ptr<xla::LocalExecutable> program;
  };

  // Wrapper for a cache entry that holds a reference to the entry until the
  // wrapper is deleted. This wrapper is the concrete type of
  // XRTCompilationCacheEntryRef returned by Lookup.
  class EntryRefImpl : public XRTCompilationCacheEntryRef {
   public:
    EntryRefImpl(XRTCompilationCache* parent, CompiledSubgraph* entry);
    ~EntryRefImpl() override;

    XRTCompilationCacheEntry get() override;

   private:
    XRTCompilationCache* parent_;  // Not owned.
    // A reference to entry_ is acquired in the contructor and released via
    // parent->DiscardEntryRef in the destructor.
    CompiledSubgraph* entry_;
  };

  // Releases one reference to entry. This is called by the cache when entry is
  // marked for eviction; or by an EntryRefImpl when it is destroyed. Before the
  // last reference to entry is released, entry is removed from cache_.
  void DiscardEntryRef(CompiledSubgraph* entry);
  void DiscardEntryRefLocked(CompiledSubgraph* entry)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Marks the oldest unmarked entry for eviction. Requires that there is at
  // least one such entry.
  void MarkOldestEntryForEviction() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Updates datastructures to indicate that entry, which had been marked for
  // eviction, has been looked up. This is called by CompileIfKeyAbsent when an
  // entry is newly created, or an entry that has been marked for eviction but
  // not yet evicted is looked up.
  //
  // First the entry is unmarked for eviction, i.e. the cache gains a reference
  // to entry, entry's last_use field is set to be the most recent value of
  // use_counter_ and entries_by_last_use_ is updated accordingly.
  //
  // Next, the size of the cache is examined to see if any other entries need to
  // be marked for eviction now that entry has been unmarked. While the total
  // number of unmarked cached entries is greater than max_cache_entries_,
  // entries are marked for eviction in LRU order. The most recently used entry
  // is never marked for eviction, so an entry larger than the max cache entries
  // will remain in the cache until it is replaced by something else.
  void LookupEntryMarkedForEviction(CompiledSubgraph* entry)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Creates a new entry by running initialize_program and places it in the
  // cache to be looked up by key. The new entry is in the 'marked for eviction'
  // state (not present in entries_by_last_use_) and the caller is expected to
  // call LookupEntryMarkedForEviction after InitializeEntry.
  //
  // **InitializeEntry releases mu_ during the call to initialize_program.**
  CompiledSubgraph* InitializeEntry(
      const string& key,
      const std::function<Status(std::unique_ptr<xla::LocalExecutable>*)>&
          initialize_program) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // The maximum number of entries that are stored in the cache before entries
  // are marked for eviction.
  const int max_cache_entries_;

  mutable absl::Mutex mu_;
  // The total number of entries that are stored and not marked for eviction.
  int cache_entries_ TF_GUARDED_BY(mu_) = 0;
  // The total number of entries that are marked for eviction.
  int marked_for_eviction_entries_ TF_GUARDED_BY(mu_) = 0;
  // The value to assign to the last_use field of the next entry that is looked
  // up.
  int64 use_counter_ TF_GUARDED_BY(mu_) = 0;
  // All the executables that can be looked up in the cache index by key. An
  // entry is marked for eviction iff it is present in cache_ and not in
  // entries_by_last_use_.
  std::unordered_map<string, CompiledSubgraph*> cache_ TF_GUARDED_BY(mu_);
  // All the executable entries that can be looked up in the cache indexed by
  // uid.
  std::unordered_map<int64, CompiledSubgraph*> entries_by_uid_
      TF_GUARDED_BY(mu_);
  // Map from last_use to entry, used to mark entries for eviction in LRU
  // order. If an entry's last_use counter is not present as a key in
  // entries_by_last_use_ then the entry has been marked for eviction.
  std::map<int64, CompiledSubgraph*> entries_by_last_use_ TF_GUARDED_BY(mu_);
};

// Looks up or create an XRTCompilationCache object within the given resource
// manager, under the default container. The max_number_of_entries sets the
// maximum number of entries within the cache (which will be LRU-evicted).
// If max_number_of_entries is set to sero, the size of the cache will be
// configured using the TF_XRT_COMPILATION_CACHE_SIZE environment variable.
xla::StatusOr<RefPtr<XRTCompilationCache>> GetOrCreateCompilationCache(
    ResourceMgr* rm, int64_t max_number_of_entries);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_COMPILATION_CACHE_H_
