// Ordered, bounded scene distances using the already installed Coal library.
// This adapter changes only the Python/C++ call boundary. It owns no geometry,
// spatial index, contact policy, solver history, or cross-state result cache.

// BEGIN exact Coal distance scans stop before any later near-contact work.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <boost/python.hpp>
#include <coal/collision_data.h>
#include <coal/collision_object.h>
#include <coal/distance.h>
#include <coal/math/transform.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
namespace bp = boost::python;
constexpr Py_ssize_t kMaximumBatchSize = 4096;

// Read the original NumPy float64 value without recomputing a transform. memcpy
// also supports a sliced or unaligned array without undefined C++ pointer loads.
double pose_value(PyArrayObject* poses, Py_ssize_t index, int row, int column) {
  const char* address = PyArray_BYTES(poses)
      + index * PyArray_STRIDE(poses, 0)
      + row * PyArray_STRIDE(poses, 1)
      + column * PyArray_STRIDE(poses, 2);
  double value;
  std::memcpy(&value, address, sizeof(value));
  return value;
}

bp::tuple scan_separated_prefix(
    const coal::CollisionGeometry& actor_shape,
    const coal::Transform3s& actor_transform,
    bp::object target_shapes,
    bp::object target_world_poses,
    const coal::DistanceRequest& request,
    double stop_distance) {
  if (!std::isfinite(stop_distance) || stop_distance < 0.) {
    throw std::invalid_argument("stop_distance must be finite and nonnegative");
  }
  const Py_ssize_t count = PySequence_Size(target_shapes.ptr());
  if (count < 0) bp::throw_error_already_set();
  if (count > kMaximumBatchSize) {
    throw std::invalid_argument("Scene distance batches may contain at most 4096 targets");
  }
  if (!PyArray_Check(target_world_poses.ptr())) {
    throw std::invalid_argument("target_world_poses must be a NumPy float64[N,4,4] array");
  }
  auto* poses = reinterpret_cast<PyArrayObject*>(target_world_poses.ptr());
  if (PyArray_NDIM(poses) != 3 || PyArray_DIM(poses, 0) != count
      || PyArray_DIM(poses, 1) != 4 || PyArray_DIM(poses, 2) != 4
      || PyArray_TYPE(poses) != NPY_DOUBLE || !PyArray_ISNOTSWAPPED(poses)) {
    throw std::invalid_argument("target_world_poses must use native float64 with shape [N,4,4]");
  }

  Py_ssize_t minimum_index = -1;
  double minimum_distance = std::numeric_limits<double>::infinity();
  coal::DistanceResult minimum_result;
  for (Py_ssize_t index = 0; index < count; ++index) {
    // Extract only the shape actually visited. In particular, an invalid later
    // entry must not raise after an earlier near item would stop Python policy.
    bp::object target = target_shapes[index];
    const coal::CollisionGeometry& target_shape =
        bp::extract<const coal::CollisionGeometry&>(target);
    coal::Matrix3s rotation;
    coal::Vec3s translation;
    for (int row = 0; row < 3; ++row) {
      translation[row] = pose_value(poses, index, row, 3);
      if (!std::isfinite(translation[row])) {
        throw std::invalid_argument("A visited target transform contains nonfinite values");
      }
      for (int column = 0; column < 3; ++column) {
        rotation(row, column) = pose_value(poses, index, row, column);
        if (!std::isfinite(rotation(row, column))) {
          throw std::invalid_argument("A visited target transform contains nonfinite values");
        }
      }
    }
    const coal::Transform3s target_transform(rotation, translation);

    // Keep exactly the raw geometry/transform overload used by query_scene.
    // A fresh result and coal::distance's fresh solver avoid ComputeDistance's
    // persistent GJK/EPA/support history even when the request uses DefaultGuess.
    coal::DistanceResult result;
    coal::distance(&actor_shape, actor_transform, &target_shape, target_transform,
                   request, result);
    const double distance = result.min_distance;
    if (!std::isfinite(distance)) {
      throw std::runtime_error("Independent scene distance returned nonfinite evidence");
    }
    if (distance <= stop_distance) {
      // Copy the native results through Coal's existing Boost.Python converter.
      // Result geometry pointers have the same borrowed lifetime as a normal
      // Coal DistanceResult; callers retain the representative shape while
      // consuming it. No target or query state is retained by this module.
      bp::object prefix = minimum_index < 0 ? bp::object() : bp::object(minimum_result);
      return bp::make_tuple(index, minimum_index, prefix, index, bp::object(result));
    }
    if (distance < minimum_distance) {
      // Strict comparison preserves the first component on equal distances,
      // including numerically identical results and differently signed zeros.
      minimum_index = index;
      minimum_distance = distance;
      minimum_result = result;
    }
  }
  bp::object prefix = minimum_index < 0 ? bp::object() : bp::object(minimum_result);
  return bp::make_tuple(count, minimum_index, prefix, -1, bp::object());
}
}  // namespace

BOOST_PYTHON_MODULE(_coal_scene_batch) {
  if (_import_array() < 0) boost::python::throw_error_already_set();
  // The installed module registers CollisionGeometry, Transform3s, request and
  // result converters. Reuse those exact types rather than defining a second
  // binding system or reconstructing geometry in this project.
  boost::python::import("coal");
  boost::python::scope().attr("MAX_BATCH_SIZE") = kMaximumBatchSize;
  boost::python::def(
      "scan_separated_prefix", &scan_separated_prefix,
      (boost::python::arg("actor_shape"), boost::python::arg("actor_transform"),
       boost::python::arg("target_shapes"), boost::python::arg("target_world_poses"),
       boost::python::arg("request"), boost::python::arg("stop_distance")),
      "Query at most 4096 targets in order using fresh Coal distance results.\n"
      "Return (separated_count, minimum_index, minimum_result, near_index, near_result).\n"
      "The first distance <= stop_distance ends the scan; later shapes and poses\n"
      "are not inspected. Missing indices/results are -1/None. Indices are local\n"
      "to this batch. Input poses must be original native float64[N,4,4] values.");
}
// END near-contact decisions and all later queries stay in Python's original order.
