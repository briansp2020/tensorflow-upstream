/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/convert/post_process_single_host_xplane.h"

#include <vector>

#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// Merges XPlanes generated by TraceMe, CUPTI API trace and Python tracer.
void MergeHostPlanesAndSortLines(XSpace* space) {
  XPlane* host_plane =
      FindOrAddMutablePlaneWithName(space, kHostThreadsPlaneName);
  std::vector<const XPlane*> additional_host_planes = FindPlanesWithNames(
      *space, {kTpuRuntimePlaneName, kCuptiDriverApiPlaneName,
               kPythonTracerPlaneName, kRoctracerApiPlaneName});
  if (!additional_host_planes.empty()) {
    MergePlanes(additional_host_planes, host_plane);
    RemovePlanes(space, additional_host_planes);
  }
  SortXLinesBy(host_plane, XLinesComparatorByName());
}

}  // namespace

void PostProcessSingleHostXSpace(XSpace* space, uint64 start_time_ns) {
  VLOG(3) << "Post processing local profiler XSpace.";
  // Post processing the collected XSpace without hold profiler lock.
  // 1. Merge all host planes and sorts lines by name.
  MergeHostPlanesAndSortLines(space);
  // 2. Normalize all timestamps by shifting timeline to profiling start time.
  // NOTE: this have to be done before sorting XSpace due to timestamp overflow.
  NormalizeTimestamps(space, start_time_ns);
  // 3. Sort each plane of the XSpace
  SortXSpace(space);
}

}  // namespace profiler
}  // namespace tensorflow
