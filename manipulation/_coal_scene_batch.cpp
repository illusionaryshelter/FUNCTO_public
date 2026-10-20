// Ordered leaf distances and conservative spatial search using installed Coal.
// Static trees own only component bounds/IDs. Query objects contain current
// support certificates, never contact policy, solver history or old distances.

// BEGIN exact Coal evidence is consumed only through the first near-contact item.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <boost/python.hpp>
#include <boost/numeric/interval.hpp>
#include <coal/BVH/BVH_model.h>
#include <coal/broadphase/detail/hierarchy_tree.h>
#include <coal/collision_data.h>
#include <coal/collision_object.h>
#include <coal/distance.h>
#include <coal/math/transform.h>
#include <coal/shape/convex.h>
#include <coal/shape/geometric_shapes.h>
#include <Eigen/QR>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

// BEGIN cache keys include the SSE arithmetic controls when this target has SSE.
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
// END status flags are excluded because ordinary arithmetic changes them.

namespace {
namespace bp = boost::python;
constexpr Py_ssize_t kMaximumBatchSize = 4096;

// BEGIN immutable component trees and per-query, outward-rounded certificates.
namespace bi = boost::numeric;
namespace bil = boost::numeric::interval_lib;
// A protected outer batch sets upward rounding once and restores it on every
// exit, including exceptions. Inside it, Boost's opposite-rounding policy uses
// exact negation for downward +,-,*,/ bounds rather than changing fenv for every
// scalar operation. Ordinary heuristics and all Coal calls stay outside guards.
using RoundingGuard = bil::save_state<bil::rounded_arith_opp<double>>;
using Interval = bi::interval<double, bil::policies<
    bil::save_state_nothing<bil::rounded_arith_opp<double>>,
    bil::checking_catch_nan<double, bil::checking_no_empty<double>>>>;
using IntervalPoint = std::array<Interval, 3>;
using Tree = coal::detail::HierarchyTree<coal::AABB>;
using Node = Tree::Node;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr std::size_t kTreeBytesPerComponent = 256;
constexpr std::size_t kQueryBytesPerComponent = 96;
constexpr std::size_t kMaximumProjectionDirections = 64;

PyArrayObject* checked_array(const bp::object& value, int type, int dimensions,
                             const char* message) {
  if (!PyArray_Check(value.ptr())) throw std::invalid_argument(message);
  auto* result = reinterpret_cast<PyArrayObject*>(value.ptr());
  if (PyArray_TYPE(result) != type || PyArray_NDIM(result) != dimensions
      || !PyArray_ISNOTSWAPPED(result)) throw std::invalid_argument(message);
  return result;
}

template <typename T>
T array_value(PyArrayObject* array, npy_intp i, npy_intp j = 0) {
  const char* address = PyArray_BYTES(array) + i * PyArray_STRIDE(array, 0);
  if (PyArray_NDIM(array) > 1) address += j * PyArray_STRIDE(array, 1);
  T value;
  std::memcpy(&value, address, sizeof(value));
  return value;
}

coal::Vec3s vector3(const bp::object& value, const char* message) {
  PyArrayObject* array = checked_array(value, NPY_DOUBLE, 1, message);
  if (PyArray_DIM(array, 0) != 3) throw std::invalid_argument(message);
  coal::Vec3s result;
  for (int axis = 0; axis < 3; ++axis) {
    result[axis] = array_value<double>(array, axis);
    if (!std::isfinite(result[axis])) throw std::invalid_argument(message);
  }
  return result;
}

bp::object index_array(const std::vector<npy_intp>& values) {
  npy_intp count = static_cast<npy_intp>(values.size());
  PyObject* result = PyArray_SimpleNew(1, &count, NPY_INTP);
  if (!result) bp::throw_error_already_set();
  if (count) std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)),
                         values.data(), values.size() * sizeof(npy_intp));
  return bp::object(bp::handle<>(result));
}

bool finite_transform(const coal::Transform3s& transform) {
  return transform.getRotation().allFinite() && transform.getTranslation().allFinite();
}

bool overlaps(const coal::AABB& a, const coal::AABB& b) {
  // These are the same closed comparisons as the original component selector.
  return (a.max_.array() >= b.min_.array()).all()
      && (a.min_.array() <= b.max_.array()).all();
}

// All interval arithmetic helpers below require a live RoundingGuard. They do
// not create one per primitive operation or per vertex in a support scan.
Interval norm_interval(const IntervalPoint& value) {
  return bi::sqrt(bi::square(value[0]) + bi::square(value[1]) + bi::square(value[2]));
}

IntervalPoint transformed_point(const IntervalPoint& point,
                                const coal::Transform3s& transform) {
  IntervalPoint result;
  for (int row = 0; row < 3; ++row) {
    result[row] = Interval(transform.getTranslation()[row]);
    for (int column = 0; column < 3; ++column)
      result[row] += Interval(transform.getRotation()(row, column)) * point[column];
  }
  return result;
}

IntervalPoint point_intervals(const coal::Vec3s& point) {
  return {Interval(point[0]), Interval(point[1]), Interval(point[2])};
}

// Bound the full original primitive's projection, not an approximate mesh of a
// cylinder. An arbitrary finite direction is valid: face normals only improve
// tightness, and none of Coal's approximate face offsets is used as a proof.
bool shape_projection(const coal::CollisionGeometry& geometry,
                      const coal::Transform3s& transform,
                      const coal::Vec3s& direction, Interval& output) {
  Interval offset(0.);
  IntervalPoint local{Interval(0.), Interval(0.), Interval(0.)};
  for (int world = 0; world < 3; ++world) {
    offset += Interval(direction[world]) * Interval(transform.getTranslation()[world]);
    for (int axis = 0; axis < 3; ++axis)
      local[axis] += Interval(direction[world]) * Interval(transform.getRotation()(world, axis));
  }
  const std::vector<coal::Vec3s>* points = nullptr;
  std::size_t point_count = 0;
  if (const auto* convex = dynamic_cast<const coal::ConvexBase*>(&geometry)) {
    if (!convex->points || convex->num_points == 0) return false;
    points = convex->points.get(); point_count = convex->num_points;
  } else if (const auto* mesh = dynamic_cast<const coal::BVHModelBase*>(&geometry)) {
    if (!mesh->vertices || mesh->num_vertices == 0) return false;
    points = mesh->vertices.get(); point_count = mesh->num_vertices;
  }
  if (points) {
    if (point_count > points->size()) return false;
    double lower = kInfinity, upper = -kInfinity;
    for (std::size_t index = 0; index < point_count; ++index) {
      const auto& point = (*points)[index];
      if (!point.allFinite()) return false;
      Interval value = offset;
      for (int axis = 0; axis < 3; ++axis) value += local[axis] * Interval(point[axis]);
      lower = std::min(lower, value.lower()); upper = std::max(upper, value.upper());
    }
    output = Interval(lower, upper);
  } else {
    Interval support(0.);
    if (const auto* box = dynamic_cast<const coal::Box*>(&geometry)) {
      if (!box->halfSide.allFinite() || (box->halfSide.array() < 0.).any()) return false;
      for (int axis = 0; axis < 3; ++axis)
        support += bi::abs(local[axis]) * Interval(box->halfSide[axis]);
    } else if (const auto* cylinder = dynamic_cast<const coal::Cylinder*>(&geometry)) {
      if (!std::isfinite(cylinder->radius) || !std::isfinite(cylinder->halfLength)
          || cylinder->radius < 0. || cylinder->halfLength < 0.) return false;
      support = Interval(cylinder->radius) * bi::sqrt(bi::square(local[0]) + bi::square(local[1]))
              + Interval(cylinder->halfLength) * bi::abs(local[2]);
    } else if (const auto* capsule = dynamic_cast<const coal::Capsule*>(&geometry)) {
      if (!std::isfinite(capsule->radius) || !std::isfinite(capsule->halfLength)
          || capsule->radius < 0. || capsule->halfLength < 0.) return false;
      support = Interval(capsule->radius) * norm_interval(local)
              + Interval(capsule->halfLength) * bi::abs(local[2]);
    } else if (const auto* sphere = dynamic_cast<const coal::Sphere*>(&geometry)) {
      if (!std::isfinite(sphere->radius) || sphere->radius < 0.) return false;
      support = Interval(sphere->radius) * norm_interval(local);
    } else {
      // No certificate for this primitive. Keeping its leaves is normal tree
      // search; the original Coal call still handles supported query shapes.
      return false;
    }
    output = offset + Interval(-support.upper(), support.upper());
  }
  if (const auto* shape = dynamic_cast<const coal::ShapeBase*>(&geometry)) {
    const double radius = shape->getSweptSphereRadius();
    if (!std::isfinite(radius) || radius < 0.) return false;
    if (radius > 0.) {
      const Interval inflation = Interval(radius) * norm_interval(local);
      output += Interval(-inflation.upper(), inflation.upper());
    }
  }
  return std::isfinite(output.lower()) && std::isfinite(output.upper());
}

struct Projection {
  IntervalPoint node_coefficients;
  Interval node_offset;
  double actor_lower, actor_upper, direction_norm_upper;
};

class SpatialQuery;

class ComponentAABBTree : public std::enable_shared_from_this<ComponentAABBTree> {
 public:
  ComponentAABBTree(const bp::object& lower_object, const bp::object& upper_object,
                    const bp::object& prunable_object) {
    const char* bounds_message = "Component bounds must be finite native float64[N,3] arrays";
    auto* lower = checked_array(lower_object, NPY_DOUBLE, 2, bounds_message);
    auto* upper = checked_array(upper_object, NPY_DOUBLE, 2, bounds_message);
    auto* prunable = checked_array(prunable_object, NPY_BOOL, 1,
                                  "prunable must be a native bool[N] array");
    const npy_intp count = PyArray_DIM(lower, 0);
    if (PyArray_DIM(lower, 1) != 3 || PyArray_DIM(upper, 1) != 3
        || PyArray_DIM(upper, 0) != count || PyArray_DIM(prunable, 0) != count)
      throw std::invalid_argument(bounds_message);
    ids_.resize(static_cast<std::size_t>(count));
    leaves_.resize(ids_.size());
    prunable_.resize(ids_.size());
    std::vector<std::unique_ptr<Node>> pending;
    std::vector<Node*> safe;
    pending.reserve(ids_.size()); safe.reserve(ids_.size());
    for (npy_intp index = 0; index < count; ++index) {
      coal::Vec3s lo, hi;
      for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = array_value<double>(lower, index, axis);
        hi[axis] = array_value<double>(upper, index, axis);
        if (!std::isfinite(lo[axis]) || !std::isfinite(hi[axis]) || lo[axis] > hi[axis])
          throw std::invalid_argument(bounds_message);
      }
      ids_[index] = index;
      auto leaf = std::make_unique<Node>();
      leaf->bv = coal::AABB(lo, hi);
      leaf->data = &ids_[index];
      leaves_[index] = leaf.get();
      prunable_[index] = array_value<npy_bool>(prunable, index) != 0;
      if (prunable_[index]) { safe.push_back(leaf.get()); pending.push_back(std::move(leaf)); }
      else { uncertified_.push_back(std::move(leaf)); }
    }
    // The maintained pointer-tree owns every node passed to init. Empty init
    // is intentionally avoided because the library's top-down builder expects
    // at least one leaf. Uncertified point/line leaves remain explicitly owned.
    if (!safe.empty()) {
      tree_.init(safe, 0);
      for (auto& leaf : pending) leaf.release();
      // This tree is immutable after construction. Coal's construction-only
      // Morton-code slot can therefore hold our node ordinal without another
      // pointer-to-index map. A query marks candidate ancestors once and never
      // traverses an unrelated part of the full field to find a local minimum.
      std::vector<Node*> stack{tree_.getRoot()};
      while (!stack.empty()) {
        Node* node = stack.back(); stack.pop_back();
        if (node_count_ > std::numeric_limits<std::uint32_t>::max())
          throw std::length_error("Component tree has too many nodes for its native node index");
        node->code = static_cast<std::uint32_t>(node_count_++);
        if (!node->isLeaf()) { stack.push_back(node->children[0]); stack.push_back(node->children[1]); }
      }
    }
  }

  std::size_t size() const { return leaves_.size(); }
  std::size_t storage_nbytes() const { return 4096 + kTreeBytesPerComponent * size(); }
  std::size_t node_count() const { return node_count_; }
  Node* root() const { return tree_.getRoot(); }
  const Node* leaf(std::size_t id) const { return leaves_[id]; }
  bool prunable(std::size_t id) const { return prunable_[id] != 0; }
  static npy_intp id(const Node* node) { return *static_cast<const npy_intp*>(node->data); }

  bp::object overlap_ids(const bp::object& lower_object, const bp::object& upper_object) const {
    coal::Vec3s lower = vector3(lower_object, "Query lower must be finite native float64[3]");
    coal::Vec3s upper = vector3(upper_object, "Query upper must be finite native float64[3]");
    if ((lower.array() > upper.array()).any()) throw std::invalid_argument("Query bounds must be ordered");
    const coal::AABB bounds(lower, upper);
    std::vector<npy_intp> selected;
    std::vector<const Node*> stack;
    if (root()) stack.push_back(root());
    while (!stack.empty()) {
      const Node* node = stack.back(); stack.pop_back();
      if (!overlaps(node->bv, bounds)) continue;
      if (node->isLeaf()) selected.push_back(id(node));
      else { stack.push_back(node->children[0]); stack.push_back(node->children[1]); }
    }
    // These leaves still use the exact original AABB candidate membership.
    // They are merely ineligible for a distance certificate until their native
    // point/segment reconstruction and composed pose have an enclosure proof.
    for (const auto& node : uncertified_)
      if (overlaps(node->bv, bounds)) selected.push_back(id(node.get()));
    std::sort(selected.begin(), selected.end());
    return index_array(selected);
  }

  std::shared_ptr<SpatialQuery> query(const coal::CollisionGeometry& actor,
      const coal::Transform3s& actor_transform, const coal::Transform3s& owner_transform,
      const bp::object& candidates, double stop_distance);

 private:
  // ids_ never changes after construction; Node::data safely points into it.
  std::vector<npy_intp> ids_;
  std::vector<Node*> leaves_;
  std::vector<unsigned char> prunable_;
  std::vector<std::unique_ptr<Node>> uncertified_;
  Tree tree_;
  std::size_t node_count_ = 0;
};

class SpatialQuery {
 public:
  SpatialQuery(std::shared_ptr<ComponentAABBTree> tree,
      const coal::CollisionGeometry& actor, const coal::Transform3s& actor_transform,
      const coal::Transform3s& owner_transform, const bp::object& candidate_object,
      double stop_distance) : tree_(std::move(tree)), stop_distance_(stop_distance) {
    if (!std::isfinite(stop_distance) || stop_distance < 0.)
      throw std::invalid_argument("stop_distance must be finite and nonnegative");
    if (!finite_transform(actor_transform) || !finite_transform(owner_transform))
      throw std::invalid_argument("Spatial query transforms must be finite");
    auto* values = checked_array(candidate_object, NPY_INTP, 1,
                                "candidate_ids must be a sorted native intp array");
    const std::size_t candidate_count = static_cast<std::size_t>(PyArray_DIM(values, 0));
    if (candidate_count > tree_->size())
      throw std::invalid_argument("candidate_ids must be unique, sorted and within the tree");
    // Reserve the exact immutable input length. Geometric growth to 2*N here
    // would overlap the seed heap's possible 48*N-byte capacity and exceed the
    // old 64*N query reserve. Exact candidate capacity also leaves room for
    // per-node cached certificates inside the current 96*N query reserve.
    candidates_.reserve(candidate_count);
    active_.assign(tree_->size(), 0);
    active_nodes_.assign(tree_->node_count(), 0);
    node_lowers_.assign(tree_->node_count(), 0.);
    node_next_directions_.assign(tree_->node_count(), 0);
    npy_intp previous = -1;
    for (npy_intp i = 0; i < PyArray_DIM(values, 0); ++i) {
      const npy_intp value = array_value<npy_intp>(values, i);
      if (value <= previous || value < 0 || static_cast<std::size_t>(value) >= tree_->size())
        throw std::invalid_argument("candidate_ids must be unique, sorted and within the tree");
      candidates_.push_back(value); active_[value] = 1; previous = value;
      if (tree_->prunable(value)) {
        const Node* node = tree_->leaf(value);
        while (node && !active_nodes_[node->code]) {
          active_nodes_[node->code] = 1;
          node = node->parent;
        }
      }
    }
    std::vector<coal::Vec3s> directions;
    const auto add_direction = [&directions](coal::Vec3s direction) {
      if (directions.size() >= kMaximumProjectionDirections) return;
      const double length = direction.norm();
      if (!direction.allFinite() || !std::isfinite(length) || length == 0.) return;
      direction /= length;  // This chooses a direction; it is not a certificate.
      for (int axis = 0; axis < 3; ++axis) if (direction[axis] != 0.) {
        if (direction[axis] < 0.) direction = -direction;
        break;
      }
      for (const auto& previous_direction : directions)
        if (direction == previous_direction) return;
      directions.push_back(direction);
    };
    // Diagonals improve circular primitive bounds when both Cartesian AABBs
    // overlap at a corner. Missing directions only weaken pruning.
    for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) add_direction(coal::Vec3s(x, y, z));
    for (int axis = 0; axis < 3; ++axis) {
      add_direction(actor_transform.getRotation().col(axis));
      add_direction(owner_transform.getRotation().col(axis));
    }
    if (const auto* convex = dynamic_cast<const coal::ConvexBase*>(&actor)) {
      if (convex->normals && !convex->normals->empty()) {
        // Stable, spread-out directions are a search heuristic only. The
        // support for each one still covers every original convex vertex;
        // this fixed limit never drops geometry, candidates, or contacts.
        const std::size_t available = kMaximumProjectionDirections - directions.size();
        const std::size_t samples = std::min(available, convex->normals->size());
        for (std::size_t sample = 0; sample < samples; ++sample) {
          const std::size_t index = sample * convex->normals->size() / samples;
          add_direction(actor_transform.getRotation() * (*convex->normals)[index]);
        }
      }
    }
    // Direction selection above uses the caller's original rounding mode.
    // Only the certificate arithmetic below needs the upward-mode batch.
    const RoundingGuard rounding;
    for (const auto& direction : directions) {
      Interval actor_projection;
      if (!shape_projection(actor, actor_transform, direction, actor_projection)) {
        projections_.clear(); break;
      }
      Projection projection;
      projection.node_offset = Interval(0.);
      projection.node_coefficients = {Interval(0.), Interval(0.), Interval(0.)};
      for (int world = 0; world < 3; ++world) {
        projection.node_offset += Interval(direction[world]) * Interval(owner_transform.getTranslation()[world]);
        for (int axis = 0; axis < 3; ++axis)
          projection.node_coefficients[axis] += Interval(direction[world])
              * Interval(owner_transform.getRotation()(world, axis));
      }
      projection.actor_lower = actor_projection.lower();
      projection.actor_upper = actor_projection.upper();
      projection.direction_norm_upper = norm_interval(point_intervals(direction)).upper();
      if (!std::isfinite(projection.direction_norm_upper) || projection.direction_norm_upper <= 0.) {
        projections_.clear(); break;
      }
      projections_.push_back(projection);
    }
  }

  std::size_t workspace_nbytes() const {
    return 4096 + kQueryBytesPerComponent * tree_->size() + kMaximumProjectionDirections * 512;
  }

  npy_intp seed_id(npy_intp prior_id = -1) {
    if (prior_id >= 0 && static_cast<std::size_t>(prior_id) < active_.size() && active_[prior_id])
      return prior_id;
    if (candidates_.empty()) return -1;
    if (projections_.empty() || !tree_->root() || !active_nodes_[tree_->root()->code])
      return candidates_.front();
    struct Entry { double lower; std::size_t serial; const Node* node; };
    struct Later {
      bool operator()(const Entry& a, const Entry& b) const {
        return a.lower != b.lower ? a.lower > b.lower : a.serial > b.serial;
      }
    };
    std::priority_queue<Entry, std::vector<Entry>, Later> queue;
    std::size_t serial = 0;
    const RoundingGuard rounding;
    queue.push({node_lower(tree_->root(), kInfinity), serial++, tree_->root()});
    while (!queue.empty()) {
      const Node* node = queue.top().node; queue.pop();
      if (node->isLeaf()) {
        const npy_intp id = ComponentAABBTree::id(node);
        if (active_[id]) return id;
      } else for (int child = 0; child < 2; ++child) {
        const Node* next = node->children[child];
        if (active_nodes_[next->code]) queue.push({node_lower(next, kInfinity), serial++, next});
      }
    }
    // An all-uncertified candidate set still supplies an exact-query seed.
    return candidates_.front();
  }

  bp::tuple retained_ids(double upper_bound) {
    if (std::isnan(upper_bound) || upper_bound < 0.)
      throw std::invalid_argument("upper_bound must be nonnegative or positive infinity");
    const double cutoff = std::max(stop_distance_, upper_bound);
    const std::size_t before = node_bound_count_;
    std::vector<npy_intp> selected;
    if (std::isinf(cutoff) || projections_.empty()) {
      selected = candidates_;
    } else {
      for (npy_intp id : candidates_) if (!tree_->prunable(id)) selected.push_back(id);
      std::vector<const Node*> stack;
      if (tree_->root() && active_nodes_[tree_->root()->code]) stack.push_back(tree_->root());
      {
        const RoundingGuard rounding;
        while (!stack.empty()) {
          const Node* node = stack.back(); stack.pop_back();
          // Strict > retains equal-distance earlier IDs. cutoff includes the
          // whole contact/measurement band, even when an incumbent is negative.
          if (node_lower(node, cutoff) > cutoff) continue;
          if (node->isLeaf()) {
            const npy_intp id = ComponentAABBTree::id(node);
            if (active_[id]) selected.push_back(id);
          } else for (int child = 0; child < 2; ++child)
            if (active_nodes_[node->children[child]->code]) stack.push_back(node->children[child]);
        }
      }
      std::sort(selected.begin(), selected.end());
    }
    bp::dict statistics;
    statistics["candidate_count"] = candidates_.size();
    statistics["retained_count"] = selected.size();
    statistics["bound_pruned_count"] = candidates_.size() - selected.size();
    statistics["node_bound_count"] = node_bound_count_ - before;
    statistics["total_node_bound_count"] = node_bound_count_;
    return bp::make_tuple(index_array(selected), statistics);
  }

 private:
  double node_lower(const Node* tree_node, double cutoff) {
    double& lower = node_lowers_[tree_node->code];
    unsigned char& next_direction = node_next_directions_[tree_node->code];
    if (lower > cutoff || next_direction == projections_.size()) return lower;
    // Count actual certificate work, not repeated visits satisfied by a cache.
    // Bounds depend only on this Query's fixed geometry and current poses, so
    // a new incumbent can reuse them. Partial early-exit certificates resume
    // if a later caller asks for a larger cutoff or a complete seed ordering.
    ++node_bound_count_;
    const coal::AABB& bounds = tree_node->bv;
    while (next_direction < projections_.size()) {
      const auto& projection = projections_[next_direction++];
      Interval node = projection.node_offset;
      for (int axis = 0; axis < 3; ++axis)
        node += projection.node_coefficients[axis] * Interval(bounds.min_[axis], bounds.max_[axis]);
      const double forward = (Interval(node.lower()) - Interval(projection.actor_upper)).lower();
      const double backward = (Interval(projection.actor_lower) - Interval(node.upper())).lower();
      const double gap = std::max(forward, backward);
      if (gap > 0.) {
        const double candidate = (Interval(gap) / Interval(projection.direction_norm_upper)).lower();
        if (std::isfinite(candidate)) lower = std::max(lower, candidate);
      }
      // The search needs a valid separating certificate, not its strongest
      // possible value. Further directions cannot invalidate this strict proof.
      if (lower > cutoff) break;
    }
    return lower;
  }

  std::shared_ptr<ComponentAABBTree> tree_;
  std::vector<npy_intp> candidates_;
  std::vector<unsigned char> active_;
  std::vector<unsigned char> active_nodes_;
  std::vector<double> node_lowers_;
  std::vector<unsigned char> node_next_directions_;
  std::vector<Projection> projections_;
  double stop_distance_;
  std::size_t node_bound_count_ = 0;
};

std::shared_ptr<SpatialQuery> ComponentAABBTree::query(
    const coal::CollisionGeometry& actor, const coal::Transform3s& actor_transform,
    const coal::Transform3s& owner_transform, const bp::object& candidates, double stop_distance) {
  return std::make_shared<SpatialQuery>(shared_from_this(), actor, actor_transform,
                                        owner_transform, candidates, stop_distance);
}

// Select nearby actual vertices, then use Eigen only to propose nonnegative
// weights. Any such weights, normalized by their exact positive sum, define a
// point inside a convex primitive. Outward interval evaluation below certifies
// that feasible point independently of least-squares accuracy or witness error.
bool convex_point(const std::vector<coal::Vec3s>& points, std::size_t count,
                  const coal::Vec3s& wanted, IntervalPoint& result) {
  if (count == 0 || count > points.size() || !wanted.allFinite()) return false;
  // Four slots bound this certificate's scratch independently of actor mesh
  // size. Every input vertex is considered; only the feasible-point proposal
  // needs the nearest few. It is not a truncation of collision geometry.
  std::array<std::pair<double, std::size_t>, 4> nearest;
  nearest.fill({kInfinity, std::numeric_limits<std::size_t>::max()});
  for (std::size_t i = 0; i < count; ++i) {
    if (!points[i].allFinite()) return false;
    const std::pair<double, std::size_t> candidate{(points[i] - wanted).squaredNorm(), i};
    for (std::size_t position = 0; position < nearest.size(); ++position) if (candidate < nearest[position]) {
      for (std::size_t move = nearest.size() - 1; move > position; --move) nearest[move] = nearest[move - 1];
      nearest[position] = candidate;
      break;
    }
  }
  const std::size_t used = std::min<std::size_t>(4, count);
  Eigen::MatrixXd coordinates(4, static_cast<Eigen::Index>(used));
  Eigen::Vector4d desired;
  desired << wanted, 1.;
  for (std::size_t i = 0; i < used; ++i) {
    coordinates.col(i).head<3>() = points[nearest[i].second]; coordinates(3, i) = 1.;
  }
  const Eigen::VectorXd proposal = coordinates.completeOrthogonalDecomposition().solve(desired);
  std::array<double, 4> weights{};
  for (std::size_t i = 0; i < used; ++i)
    weights[i] = std::isfinite(proposal[i]) ? std::max(0., proposal[i]) : 0.;
  // A failed/ill-conditioned proposal still has a valid nearest-vertex point.
  if (std::all_of(weights.begin(), weights.end(), [](double x) { return x == 0.; })) weights[0] = 1.;
  const RoundingGuard rounding;
  Interval denominator(0.);
  result = {Interval(0.), Interval(0.), Interval(0.)};
  for (std::size_t i = 0; i < used; ++i) {
    denominator += Interval(weights[i]);
    for (int axis = 0; axis < 3; ++axis)
      result[axis] += Interval(weights[i]) * Interval(points[nearest[i].second][axis]);
  }
  if (!std::isfinite(denominator.upper()) || denominator.lower() <= 0.) return false;
  for (int axis = 0; axis < 3; ++axis) result[axis] /= denominator;
  return true;
}

bool feasible_local_point(const coal::CollisionGeometry& geometry, const coal::Vec3s& wanted,
                          int triangle_id, IntervalPoint& point) {
  if (!wanted.allFinite()) return false;
  if (const auto* convex = dynamic_cast<const coal::ConvexBase*>(&geometry)) {
    return convex->points && convex_point(*convex->points, convex->num_points, wanted, point);
  }
  if (const auto* mesh = dynamic_cast<const coal::BVHModelBase*>(&geometry)) {
    if (!mesh->vertices || !mesh->tri_indices || mesh->num_tris == 0) return false;
    // Combining different non-coplanar mesh triangles would invent a point in
    // their convex hull. Use one real triangle, preferably Coal's witness ID.
    const std::size_t index = triangle_id >= 0 && static_cast<unsigned>(triangle_id) < mesh->num_tris
        ? static_cast<std::size_t>(triangle_id) : 0;
    if (index >= mesh->tri_indices->size()) return false;
    const coal::Triangle& triangle = (*mesh->tri_indices)[index];
    std::vector<coal::Vec3s> vertices;
    for (int corner = 0; corner < 3; ++corner) {
      const std::size_t vertex = triangle[corner];
      if (vertex >= mesh->vertices->size()) return false;
      vertices.push_back((*mesh->vertices)[vertex]);
    }
    return convex_point(vertices, vertices.size(), wanted, point);
  }
  if (const auto* box = dynamic_cast<const coal::Box*>(&geometry)) {
    if (!box->halfSide.allFinite() || (box->halfSide.array() < 0.).any()) return false;
    for (int axis = 0; axis < 3; ++axis)
      point[axis] = Interval(std::max(-box->halfSide[axis], std::min(box->halfSide[axis], wanted[axis])));
    return true;
  }
  double radius = 0., half_length = 0.;
  bool cylinder = false;
  if (const auto* shape = dynamic_cast<const coal::Cylinder*>(&geometry)) {
    radius = shape->radius; half_length = shape->halfLength; cylinder = true;
  } else if (const auto* shape = dynamic_cast<const coal::Capsule*>(&geometry)) {
    radius = shape->radius; half_length = shape->halfLength;
  } else if (const auto* shape = dynamic_cast<const coal::Sphere*>(&geometry)) {
    radius = shape->radius;
  } else return false;
  if (!std::isfinite(radius) || !std::isfinite(half_length) || radius < 0. || half_length < 0.) return false;
  const double z = std::max(-half_length, std::min(half_length, wanted[2]));
  point = {Interval(0.), Interval(0.), Interval(z)};
  if (radius == 0.) return true;
  // The exact radial vector, normalized by its exact norm, is on the original
  // circle/sphere. Independent interval coordinates enclose that same point.
  coal::Vec3s direction(wanted[0], wanted[1], cylinder ? 0. : wanted[2] - z);
  if (direction.isZero(0.)) return true;
  const RoundingGuard rounding;
  IntervalPoint radial = point_intervals(direction);
  const Interval length = norm_interval(radial);
  if (!std::isfinite(length.upper()) || length.lower() <= 0.) return false;
  if (cylinder && length.upper() <= radius) {
    // A cap witness inside the disk need not be moved to its rim. This point
    // is certified inside the original circular cross-section already.
    point[0] = radial[0]; point[1] = radial[1];
    return true;
  }
  for (int axis = 0; axis < 3; ++axis) point[axis] += Interval(radius) * radial[axis] / length;
  return true;
}

// BEGIN actual hull faces improve a feasible proposal without changing geometry.
bool triangle_convex_point(const coal::CollisionGeometry& geometry,
                          const coal::Vec3s& wanted, IntervalPoint& point) {
  const auto* convex = dynamic_cast<const coal::Convex<coal::Triangle>*>(&geometry);
  if (!convex || !wanted.allFinite() || !convex->points || !convex->polygons
      || convex->num_points == 0 || convex->num_points > convex->points->size()
      || convex->num_polygons == 0 || convex->num_polygons > convex->polygons->size())
    return false;

  // The old nearest-four-vertices proposal can select the opposite small face
  // when the real witness lies inside a large face. Enumerate the native hull's
  // actual triangles instead. Eigen chooses barycentric weights only; neither
  // the projection accuracy nor a face normal/offset participates in the proof.
  // Scratch remains three vertices and two small fixed matrices, independently
  // of hull size. All proposal arithmetic uses the caller's rounding mode.
  std::array<std::size_t, 3> selected{};
  std::array<double, 3> selected_weights{};
  double best_squared_distance = kInfinity;
  bool found = false;
  for (std::size_t face = 0; face < convex->num_polygons; ++face) {
    const coal::Triangle& triangle = (*convex->polygons)[face];
    std::array<std::size_t, 3> indices{};
    for (int corner = 0; corner < 3; ++corner) {
      indices[corner] = triangle[corner];
      if (indices[corner] >= convex->num_points
          || !(*convex->points)[indices[corner]].allFinite()) return false;
    }
    const coal::Vec3s& origin = (*convex->points)[indices[0]];
    Eigen::Matrix<double, 3, 2> edges;
    edges.col(0) = (*convex->points)[indices[1]] - origin;
    edges.col(1) = (*convex->points)[indices[2]] - origin;
    if (!edges.allFinite()) continue;
    const coal::Vec3s desired = wanted - origin;
    if (!desired.allFinite()) continue;
    const Eigen::Vector2d proposal = edges.completeOrthogonalDecomposition().solve(desired);
    if (!proposal.allFinite()) continue;
    std::array<double, 3> weights{
        std::max(0., 1. - proposal[0] - proposal[1]),
        std::max(0., proposal[0]), std::max(0., proposal[1])};
    const double denominator = weights[0] + weights[1] + weights[2];
    if (!std::isfinite(denominator) || denominator <= 0.) continue;
    coal::Vec3s candidate = coal::Vec3s::Zero();
    for (int corner = 0; corner < 3; ++corner)
      candidate += weights[corner] * (*convex->points)[indices[corner]];
    candidate /= denominator;
    const double squared_distance = (candidate - wanted).squaredNorm();
    if (std::isfinite(squared_distance) && squared_distance < best_squared_distance) {
      selected = indices;
      selected_weights = weights;
      best_squared_distance = squared_distance;
      found = true;
    }
  }
  if (!found) return false;

  // Certify only the selected proposal. Nonnegative finite binary64 weights
  // divided by their exact positive sum define a point on this real triangle.
  // Interval evaluation encloses that point even for a rank-deficient face or
  // a Coal witness outside the hull; no approximate projected coordinate is
  // admitted as a feasible point directly.
  const RoundingGuard rounding;
  Interval denominator(0.);
  point = {Interval(0.), Interval(0.), Interval(0.)};
  for (int corner = 0; corner < 3; ++corner) {
    denominator += Interval(selected_weights[corner]);
    for (int axis = 0; axis < 3; ++axis)
      point[axis] += Interval(selected_weights[corner]) * Interval((*convex->points)[selected[corner]][axis]);
  }
  if (!std::isfinite(denominator.upper()) || denominator.lower() <= 0.) return false;
  for (int axis = 0; axis < 3; ++axis) point[axis] /= denominator;
  return true;
}

double feasible_pair_upper(IntervalPoint first, const coal::Transform3s& first_transform,
                           IntervalPoint second, const coal::Transform3s& second_transform) {
  const RoundingGuard rounding;
  first = transformed_point(first, first_transform);
  second = transformed_point(second, second_transform);
  IntervalPoint difference;
  for (int axis = 0; axis < 3; ++axis) difference[axis] = first[axis] - second[axis];
  const double upper = norm_interval(difference).upper();
  return std::isfinite(upper) && upper >= 0. ? upper : kInfinity;
}
// END one certified triangle proposal is independent of raw Coal evidence.

double certified_distance_upper_bound(const coal::CollisionGeometry& actor,
    const coal::Transform3s& actor_transform, const coal::CollisionGeometry& target,
    const coal::Transform3s& target_transform, const coal::DistanceResult& result) {
  if (!finite_transform(actor_transform) || !finite_transform(target_transform)) return kInfinity;
  // Transpose transforms only choose desired local points; their rounding and
  // approximate orthogonality cannot invalidate the subsequent feasible-point
  // construction. Actual world points use the original matrices in intervals.
  const coal::Vec3s first_wanted = actor_transform.getRotation().transpose()
      * (result.nearest_points[0] - actor_transform.getTranslation());
  const coal::Vec3s second_wanted = target_transform.getRotation().transpose()
      * (result.nearest_points[1] - target_transform.getTranslation());
  IntervalPoint first, second;
  if (!feasible_local_point(actor, first_wanted, result.b1, first)
      || !feasible_local_point(target, second_wanted, result.b2, second)) return kInfinity;
  // BEGIN retain the generic certificate and accept only certified tightening.
  // A worse/invalid triangle proposal cannot weaken the old upper bound. Each
  // alternative is a distance between actual points in the original solids,
  // so the smallest of these upper bounds is still an upper bound. Comparing
  // mixed endpoint proposals also avoids assuming that two independently
  // nearer-to-witness points necessarily give a shorter pair distance.
  double upper = feasible_pair_upper(first, actor_transform, second, target_transform);
  IntervalPoint triangle_first, triangle_second;
  const bool first_triangle = triangle_convex_point(actor, first_wanted, triangle_first);
  const bool second_triangle = triangle_convex_point(target, second_wanted, triangle_second);
  if (first_triangle)
    upper = std::min(upper, feasible_pair_upper(triangle_first, actor_transform, second, target_transform));
  if (second_triangle)
    upper = std::min(upper, feasible_pair_upper(first, actor_transform, triangle_second, target_transform));
  if (first_triangle && second_triangle)
    upper = std::min(upper, feasible_pair_upper(triangle_first, actor_transform, triangle_second, target_transform));
  return upper;
  // END no caller observes a changed distance, witness, solver, or contact band.
}
// END certificates affect search only; original Coal results remain untouched.

// BEGIN expose existing native support and feasible-point enclosures to raw TSDF queries.
bp::tuple native_projection_bounds(const coal::CollisionGeometry& geometry,
    const coal::Transform3s& transform, const bp::object& direction_object) {
  const coal::Vec3s direction = vector3(direction_object, "direction must be a finite float64 3-vector");
  if (!finite_transform(transform) || direction.isZero(0.))
    throw std::invalid_argument("Native projection requires a finite transform and nonzero direction");
  const RoundingGuard rounding;
  Interval projection;
  if (!shape_projection(geometry, transform, direction, projection))
    throw std::invalid_argument("The registered native shape has no supported projection enclosure");
  return bp::make_tuple(projection.lower(), projection.upper());
}

bp::tuple point_bounds(const IntervalPoint& point) {
  bp::list lower, upper;
  for (int axis = 0; axis < 3; ++axis) {
    lower.append(point[axis].lower());
    upper.append(point[axis].upper());
  }
  return bp::make_tuple(lower, upper);
}

// BEGIN construct interior as well as surface points of the registered convex.
bool volume_convex_point(const coal::CollisionGeometry& geometry,
                         const coal::Vec3s& wanted, IntervalPoint& point) {
  const auto* convex = dynamic_cast<const coal::Convex<coal::Triangle>*>(&geometry);
  if (!convex || !convex->points || !convex->polygons || convex->num_points == 0
      || convex->num_points > convex->points->size() || convex->num_polygons == 0
      || convex->num_polygons > convex->polygons->size() || !wanted.allFinite()) return false;
  // The exact mean of all original vertices belongs to the convex entity.
  // Its rounded coordinate only proposes weights. Certification below uses
  // the interval sum of the original vertices, not this rounded mean.
  coal::Vec3s center = coal::Vec3s::Zero();
  for (std::size_t i = 0; i < convex->num_points; ++i) {
    if (!(*convex->points)[i].allFinite()) return false;
    center += (*convex->points)[i];
  }
  center /= static_cast<double>(convex->num_points);
  if (!center.allFinite()) return false;
  std::array<std::size_t, 3> selected{};
  std::array<double, 4> selected_weights{};
  double best = kInfinity;
  bool found = false;
  for (std::size_t face = 0; face < convex->num_polygons; ++face) {
    const coal::Triangle& triangle = (*convex->polygons)[face];
    Eigen::Matrix3d edges;
    for (int i = 0; i < 3; ++i) {
      if (triangle[i] >= convex->num_points) return false;
      edges.col(i) = (*convex->points)[triangle[i]] - center;
    }
    const coal::Vec3s proposal = edges.completeOrthogonalDecomposition().solve(wanted - center);
    if (!proposal.allFinite()) continue;
    std::array<double, 4> weights{std::max(0., 1. - proposal.sum()),
        std::max(0., proposal[0]), std::max(0., proposal[1]), std::max(0., proposal[2])};
    const double denominator = weights[0] + weights[1] + weights[2] + weights[3];
    if (!std::isfinite(denominator) || denominator <= 0.) continue;
    coal::Vec3s candidate = weights[0] * center;
    for (int i = 0; i < 3; ++i) candidate += weights[i+1] * (*convex->points)[triangle[i]];
    candidate /= denominator;
    const double distance = (candidate - wanted).squaredNorm();
    if (std::isfinite(distance) && distance < best) {
      best = distance; found = true; selected_weights = weights;
      for (int i = 0; i < 3; ++i) selected[i] = triangle[i];
    }
  }
  if (!found) return false;
  // Only this feasible point is constructed. The registered shape and its
  // faces remain unchanged; no tetrahedral collision model is created.
  const RoundingGuard rounding;
  IntervalPoint mean{Interval(0.), Interval(0.), Interval(0.)};
  for (std::size_t i = 0; i < convex->num_points; ++i)
    for (int axis = 0; axis < 3; ++axis) mean[axis] += Interval((*convex->points)[i][axis]);
  for (int axis = 0; axis < 3; ++axis) mean[axis] /= Interval(static_cast<double>(convex->num_points));
  Interval denominator(selected_weights[0]);
  for (int axis = 0; axis < 3; ++axis) point[axis] = Interval(selected_weights[0]) * mean[axis];
  for (int i = 0; i < 3; ++i) {
    denominator += Interval(selected_weights[i+1]);
    for (int axis = 0; axis < 3; ++axis)
      point[axis] += Interval(selected_weights[i+1]) * Interval((*convex->points)[selected[i]][axis]);
  }
  if (denominator.lower() <= 0. || !std::isfinite(denominator.upper())) return false;
  for (int axis = 0; axis < 3; ++axis) point[axis] /= denominator;
  return true;
}
// END the same original convex entity supplies feasible points throughout its volume.

bp::dict native_feasible_point_bounds(const coal::CollisionGeometry& geometry,
    const coal::Transform3s& transform, const bp::object& wanted_object) {
  const coal::Vec3s wanted_world = vector3(wanted_object, "wanted_world must be a finite float64 3-vector");
  if (!finite_transform(transform))
    throw std::invalid_argument("Native point construction requires a finite transform");
  // This inverse chooses a proposal only. Membership follows from the existing
  // nonnegative vertex combination or analytic primitive construction.
  const coal::Vec3s wanted_local = transform.getRotation().transpose()
      * (wanted_world - transform.getTranslation());
  IntervalPoint local;
  const bool is_convex = dynamic_cast<const coal::ConvexBase*>(&geometry) != nullptr;
  if (!(is_convex ? volume_convex_point(geometry, wanted_local, local)
                  : feasible_local_point(geometry, wanted_local, -1, local)))
    throw std::invalid_argument("The registered native shape has no supported feasible-point construction");
  coal::Transform3s identity;
  identity.setIdentity();
  const IntervalPoint wanted = point_intervals(wanted_world);
  double upper = feasible_pair_upper(local, transform, wanted, identity);
  IntervalPoint triangle;
  if (triangle_convex_point(geometry, wanted_local, triangle)) {
    const double candidate_upper = feasible_pair_upper(triangle, transform, wanted, identity);
    if (candidate_upper < upper) {
      local = triangle;
      upper = candidate_upper;
    }
  }
  if (!std::isfinite(upper))
    throw std::runtime_error("Native feasible-point enclosure is nonfinite");
  // Keep the chosen local/world intervals together. A raw Coal nearest point
  // need not equal this feasible point and is never substituted afterwards.
  const RoundingGuard rounding;
  bp::dict result;
  result["local_bounds"] = point_bounds(local);
  result["world_bounds"] = point_bounds(transformed_point(local, transform));
  return result;
}
// END the bridge exports original-entity evidence, not task geometry or policy.

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

// BEGIN independent raw distances preserve input bits and indexed consumption.
// Python ownership and error objects are created/destroyed with the GIL held.
// Workers only read copied transforms/native pointers and write their own slot.
struct OrderedDistanceEntry {
  bp::object first_owner, second_owner;
  bp::object error_type, error_value, error_traceback;
  const coal::CollisionGeometry* first = nullptr;
  const coal::CollisionGeometry* second = nullptr;
  coal::Transform3s first_transform, second_transform;
  coal::DistanceResult distance;
  std::exception_ptr native_error;
  bool prepared = false;
};

class ReleasedGIL {
 public:
  ReleasedGIL() : state_(PyEval_SaveThread()) {}
  ~ReleasedGIL() { PyEval_RestoreThread(state_); }
  ReleasedGIL(const ReleasedGIL&) = delete;
  ReleasedGIL& operator=(const ReleasedGIL&) = delete;
 private:
  PyThreadState* state_;
};

class QueryFloatingEnvironment {
 public:
  explicit QueryFloatingEnvironment(const std::fenv_t& requested) {
    if (std::fegetenv(&previous_) != 0)
      throw std::runtime_error("Cannot save distance worker floating-point environment");
    if (std::fesetenv(&requested) != 0) {
      std::fesetenv(&previous_);
      throw std::runtime_error("Cannot inherit distance caller floating-point environment");
    }
  }
  ~QueryFloatingEnvironment() { std::fesetenv(&previous_); }
  QueryFloatingEnvironment(const QueryFloatingEnvironment&) = delete;
  QueryFloatingEnvironment& operator=(const QueryFloatingEnvironment&) = delete;
 private:
  std::fenv_t previous_;
};

class OrderedDistanceBatch {
 public:
  OrderedDistanceBatch(bp::object first_shapes, bp::object first_transforms,
      bp::object second_shapes, bp::object second_transforms,
      const coal::DistanceRequest& request, int workers) : request_(request) {
    // BEGIN the caller participates in the explicitly authorized eight-worker limit.
    if (workers < 1 || workers > 8)
      throw std::invalid_argument("Distance workers must be between 1 and 8, including the caller");
    // END scheduling, request isolation and result consumption are unchanged.
    // A shared request with history would make later inputs depend on earlier
    // calls. Copying it per slot is deliberately forbidden for this interface.
    COAL_COMPILER_DIAGNOSTIC_PUSH
    COAL_COMPILER_DIAGNOSTIC_IGNORED_DEPRECECATED_DECLARATIONS
    const bool stateful = request.gjk_initial_guess == coal::GJKInitialGuess::CachedGuess
        || request.enable_cached_gjk_guess;
    COAL_COMPILER_DIAGNOSTIC_POP
    if (stateful)
      throw std::invalid_argument("Independent distance batches require requests without cached GJK history");
    const Py_ssize_t count = PySequence_Size(first_shapes.ptr());
    if (count < 0) bp::throw_error_already_set();
    for (const auto* sequence : {&first_transforms, &second_shapes, &second_transforms}) {
      const Py_ssize_t length = PySequence_Size(sequence->ptr());
      if (length < 0) bp::throw_error_already_set();
      if (length != count)
        throw std::invalid_argument("Distance batch shape and transform sequences must have equal lengths");
    }
    entries_.resize(static_cast<std::size_t>(count));
    worker_count_ = std::min<std::size_t>(static_cast<std::size_t>(workers), entries_.size());
    for (Py_ssize_t index = 0; index < count; ++index) {
      auto& entry = entries_[static_cast<std::size_t>(index)];
      try {
        // Keep direct owners, rather than merely retaining a mutable input list.
        // Transform copies invoke no FK, matrix product or normalization.
        entry.first_owner = first_shapes[index];
        entry.first = &bp::extract<const coal::CollisionGeometry&>(entry.first_owner)();
        entry.first_transform = bp::extract<const coal::Transform3s&>(first_transforms[index]);
        entry.second_owner = second_shapes[index];
        entry.second = &bp::extract<const coal::CollisionGeometry&>(entry.second_owner)();
        entry.second_transform = bp::extract<const coal::Transform3s&>(second_transforms[index]);
        entry.prepared = true;
      } catch (const bp::error_already_set&) {
        // A later malformed descriptor must not preempt an earlier caller-side
        // component/contact failure. Preserve its Python exception until result().
        PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        entry.error_type = type ? bp::object(bp::handle<>(type)) : bp::object();
        entry.error_value = value ? bp::object(bp::handle<>(value)) : bp::object();
        entry.error_traceback = traceback ? bp::object(bp::handle<>(traceback)) : bp::object();
      } catch (...) {
        entry.native_error = std::current_exception();
      }
    }
    if (entries_.empty()) return;
    std::fenv_t caller_environment;
    if (std::fegetenv(&caller_environment) != 0)
      throw std::runtime_error("Cannot read distance caller floating-point environment");
    {
      const ReleasedGIL released;
      const auto run = [this, &caller_environment](std::size_t lane) noexcept {
        try {
          const QueryFloatingEnvironment environment(caller_environment);
          for (std::size_t index = lane; index < entries_.size(); index += worker_count_) {
            auto& entry = entries_[index];
            if (!entry.prepared) continue;
            try {
              // Each item has exactly the caller's initial request and a cold
              // solver, even when a preceding item used the same shape pair.
              // No original request, shape, shared result or solver is mutated.
              const coal::DistanceRequest local_request(request_);
              coal::distance(entry.first, entry.first_transform, entry.second,
                             entry.second_transform, local_request, entry.distance);
            } catch (...) {
              entry.native_error = std::current_exception();
            }
          }
        } catch (...) {
          const std::exception_ptr error = std::current_exception();
          for (std::size_t index = lane; index < entries_.size(); index += worker_count_)
            if (entries_[index].prepared) entries_[index].native_error = error;
        }
      };
      std::vector<std::thread> threads;
      threads.reserve(worker_count_ - 1);
      try {
        for (std::size_t lane = 1; lane < worker_count_; ++lane) threads.emplace_back(run, lane);
        run(0);  // The caller is one worker, so total concurrency never exceeds 8.
      } catch (...) {
        // A thread creation failure must never destroy a joinable thread or
        // return while a worker still refers to the batch under construction.
        for (auto& thread : threads) thread.join();
        throw;
      }
      for (auto& thread : threads) thread.join();
    }
  }

  bp::object result(Py_ssize_t index) const {
    if (index < 0 || static_cast<std::size_t>(index) >= entries_.size())
      throw std::out_of_range("Distance batch result index is out of range");
    const auto& entry = entries_[static_cast<std::size_t>(index)];
    if (entry.error_type.ptr() != Py_None) {
      PyObject* type = entry.error_type.ptr();
      PyObject* value = entry.error_value.ptr() == Py_None ? nullptr : entry.error_value.ptr();
      PyObject* traceback = entry.error_traceback.ptr() == Py_None ? nullptr : entry.error_traceback.ptr();
      Py_INCREF(type); Py_XINCREF(value); Py_XINCREF(traceback);
      PyErr_Restore(type, value, traceback);
      bp::throw_error_already_set();
    }
    if (entry.native_error) std::rethrow_exception(entry.native_error);
    // A value copy keeps repeated alias consumption and earlier Python results
    // independent. Geometry pointers have Coal's usual borrowed-shape lifetime;
    // this batch owns every shape until the caller has consumed its results.
    return bp::object(entry.distance);
  }

  std::size_t size() const { return entries_.size(); }
  std::size_t worker_count() const { return worker_count_; }
  std::size_t storage_nbytes() const {
    // Native value storage only. Existing shape payloads, Python referents,
    // exception allocations and transient thread/solver work are not counted.
    return sizeof(*this) + entries_.capacity() * sizeof(OrderedDistanceEntry);
  }

 private:
  coal::DistanceRequest request_;
  std::vector<OrderedDistanceEntry> entries_;
  std::size_t worker_count_ = 0;
};

std::shared_ptr<OrderedDistanceBatch> ordered_distances(
    bp::object first_shapes, bp::object first_transforms,
    bp::object second_shapes, bp::object second_transforms,
    const coal::DistanceRequest& request, int workers) {
  return std::make_shared<OrderedDistanceBatch>(first_shapes, first_transforms,
      second_shapes, second_transforms, request, workers);
}
// END independent batches never own scene policy, spatial hints or cache writes.

// BEGIN stable component ownership and prepared original-world-pose batches.
// Every registered slot becomes immutable. An evicted Python pose-cache entry
// remains valid while a generator or batch holds its prepared object alive.
struct ComponentShapeEntry {
  bp::object owner;
  const coal::CollisionGeometry* shape = nullptr;
};
static_assert(sizeof(ComponentShapeEntry) <= 128,
              "Update the Python component-table reservation for this ABI");

class ComponentTable {
 public:
  explicit ComponentTable(Py_ssize_t count) {
    if (count < 0) throw std::invalid_argument("Component table size must be nonnegative");
    entries_.resize(static_cast<std::size_t>(count));
  }

  const coal::CollisionGeometry* register_shape(std::size_t index, const bp::object& owner) {
    auto& entry = entries_.at(index);
    if (entry.shape) {
      // A flat ID belongs to exactly one geometry for this source map. Existing
      // entries bypass Boost extraction, but never silently accept a new shape.
      if (entry.owner.ptr() != owner.ptr())
        throw std::invalid_argument("A registered component ID cannot change its native shape");
      return entry.shape;
    }
    const auto* shape = &bp::extract<const coal::CollisionGeometry&>(owner)();
    entry.owner = owner;
    entry.shape = shape;
    return shape;
  }

  std::size_t size() const { return entries_.size(); }
  std::size_t slot_nbytes() const { return sizeof(ComponentShapeEntry); }
  std::size_t storage_nbytes() const {
    return sizeof(*this) + entries_.capacity() * sizeof(ComponentShapeEntry);
  }

 private:
  std::vector<ComponentShapeEntry> entries_;
};

// Both callers use source-order, bounded arrays. Validation happens while the
// GIL is held; workers never read a caller-mutable NumPy descriptor.
std::vector<npy_intp> component_ids(const bp::object& object, std::size_t capacity) {
  auto* values = checked_array(object, NPY_INTP, 1,
      "Component IDs must be a sorted native intp array");
  const npy_intp count = PyArray_DIM(values, 0);
  if (count > kMaximumBatchSize)
    throw std::invalid_argument("Scene distance batches may contain at most 4096 targets");
  std::vector<npy_intp> ids;
  ids.reserve(static_cast<std::size_t>(count));
  npy_intp previous = -1;
  for (npy_intp index = 0; index < count; ++index) {
    const npy_intp value = array_value<npy_intp>(values, index);
    if (value <= previous || value < 0 || static_cast<std::size_t>(value) >= capacity)
      throw std::invalid_argument("Component IDs must be unique, sorted and within their table");
    ids.push_back(value);
    previous = value;
  }
  return ids;
}

struct DeferredComponentError {
  bp::object type, value, traceback;
  std::exception_ptr native;

  void capture_python() {
    PyObject *error_type = nullptr, *error_value = nullptr, *error_traceback = nullptr;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    type = error_type ? bp::object(bp::handle<>(error_type)) : bp::object();
    value = error_value ? bp::object(bp::handle<>(error_value)) : bp::object();
    traceback = error_traceback ? bp::object(bp::handle<>(error_traceback)) : bp::object();
  }

  void raise() const {
    if (type.ptr() != Py_None) {
      PyObject* error_type = type.ptr();
      PyObject* error_value = value.ptr() == Py_None ? nullptr : value.ptr();
      PyObject* error_traceback = traceback.ptr() == Py_None ? nullptr : traceback.ptr();
      Py_INCREF(error_type); Py_XINCREF(error_value); Py_XINCREF(error_traceback);
      PyErr_Restore(error_type, error_value, error_traceback);
      bp::throw_error_already_set();
    }
    if (native) std::rethrow_exception(native);
  }
};

struct PreparedComponentEntry {
  std::array<double, 16> original_pose{};
  coal::Transform3s transform;
  const coal::CollisionGeometry* shape = nullptr;
  DeferredComponentError error;
  bool initialized = false;
  bool ready = false;
};
static_assert(sizeof(PreparedComponentEntry) <= 512,
              "Update the Python prepared-pose reservation for this ABI");

class ComponentDistanceBatch;

class PreparedComponents : public std::enable_shared_from_this<PreparedComponents> {
 public:
  explicit PreparedComponents(std::shared_ptr<ComponentTable> table) : table_(std::move(table)) {
    if (!table_) throw std::invalid_argument("Prepared components require their source table");
    entries_.resize(table_->size());
  }

  bp::object missing(const bp::object& object) const {
    const auto ids = component_ids(object, size());
    std::vector<npy_intp> selected;
    selected.reserve(ids.size());
    for (const auto id : ids)
      if (!entries_[static_cast<std::size_t>(id)].initialized) selected.push_back(id);
    npy_intp dimensions[] = {static_cast<npy_intp>(selected.size())};
    auto* array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dimensions, NPY_INTP));
    if (!array) bp::throw_error_already_set();
    if (!selected.empty())
      std::memcpy(PyArray_DATA(array), selected.data(), selected.size() * sizeof(npy_intp));
    return bp::object(bp::handle<>(reinterpret_cast<PyObject*>(array)));
  }

  void register_components(const bp::object& object, const bp::object& shapes,
                           const bp::object& world_poses) {
    const auto ids = component_ids(object, size());
    const Py_ssize_t count = PySequence_Size(shapes.ptr());
    if (count < 0) bp::throw_error_already_set();
    auto* poses = checked_array(world_poses, NPY_DOUBLE, 3,
        "Component world poses must be native float64[N,4,4]");
    if (static_cast<std::size_t>(count) != ids.size() || PyArray_DIM(poses, 0) != count
        || PyArray_DIM(poses, 1) != 4 || PyArray_DIM(poses, 2) != 4)
      throw std::invalid_argument("Component IDs, shapes and float64[N,4,4] poses must align");
    // Global descriptor failures above belong to this registration call. A
    // particular shape/pose failure below belongs only to that source ID and
    // must not override an earlier near-contact decision.
    for (std::size_t index = 0; index < ids.size(); ++index) {
      auto& entry = entries_[static_cast<std::size_t>(ids[index])];
      if (entry.initialized)
        throw std::invalid_argument("Prepared component slots cannot be registered twice");
    }
    for (std::size_t index = 0; index < ids.size(); ++index) {
      auto& entry = entries_[static_cast<std::size_t>(ids[index])];
      entry.initialized = true;
      try {
        entry.shape = table_->register_shape(static_cast<std::size_t>(ids[index]),
                                            shapes[static_cast<Py_ssize_t>(index)]);
        coal::Matrix3s rotation;
        coal::Vec3s translation;
        for (int row = 0; row < 4; ++row) {
          for (int column = 0; column < 4; ++column) {
            const double value = pose_value(poses, static_cast<Py_ssize_t>(index), row, column);
            entry.original_pose[static_cast<std::size_t>(row * 4 + column)] = value;
            // Coal reads the first three rows. Preserve even unused bottom-row
            // bits rather than regenerating a homogeneous matrix on retrieval.
            if (row < 3) {
              if (!std::isfinite(value))
                throw std::invalid_argument("A visited target transform contains nonfinite values");
              if (column < 3) rotation(row, column) = value;
              else translation[row] = value;
            }
          }
        }
        entry.transform = coal::Transform3s(rotation, translation);
        entry.ready = true;
      } catch (const bp::error_already_set&) {
        entry.error.capture_python();
      } catch (...) {
        entry.error.native = std::current_exception();
      }
    }
  }

  bp::object pose(Py_ssize_t index) const {
    const auto& entry = checked_entry(index);
    entry.error.raise();
    npy_intp dimensions[] = {4, 4};
    auto* array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, dimensions, NPY_DOUBLE));
    if (!array) bp::throw_error_already_set();
    std::memcpy(PyArray_DATA(array), entry.original_pose.data(), sizeof(double) * 16);
    return bp::object(bp::handle<>(reinterpret_cast<PyObject*>(array)));
  }

  const PreparedComponentEntry& checked_entry(Py_ssize_t index) const {
    if (index < 0 || static_cast<std::size_t>(index) >= size())
      throw std::out_of_range("Prepared component ID is out of range");
    const auto& entry = entries_[static_cast<std::size_t>(index)];
    if (!entry.initialized)
      throw std::invalid_argument("A required component has not been prepared");
    return entry;
  }

  std::shared_ptr<ComponentDistanceBatch> scan(const bp::object& actor,
      const coal::Transform3s& actor_transform, const bp::object& ids,
      const coal::DistanceRequest& request, int workers);

  std::size_t size() const { return entries_.size(); }
  std::size_t slot_nbytes() const { return sizeof(PreparedComponentEntry); }
  std::size_t storage_nbytes() const {
    // The separately reported table owns shapes once, independent of poses.
    return sizeof(*this) + entries_.capacity() * sizeof(PreparedComponentEntry);
  }

 private:
  std::shared_ptr<ComponentTable> table_;
  std::vector<PreparedComponentEntry> entries_;
};

struct ComponentDistanceEntry {
  const PreparedComponentEntry* target = nullptr;
  coal::DistanceResult distance;
  std::exception_ptr error;
  bool queried = false;
};
static_assert(sizeof(ComponentDistanceEntry) <= 1024,
              "Update the Python component-batch reservation for this ABI");

class ComponentDistanceBatch {
 public:
  ComponentDistanceBatch(std::shared_ptr<PreparedComponents> prepared, const bp::object& actor,
      const coal::Transform3s& actor_transform, const bp::object& object,
      const coal::DistanceRequest& request, int workers)
      : prepared_(std::move(prepared)), actor_owner_(actor), actor_transform_(actor_transform),
        request_(request) {
    // BEGIN component work shares the explicitly authorized eight-worker limit.
    if (workers < 1 || workers > 8)
      throw std::invalid_argument("Distance workers must be between 1 and 8, including the caller");
    // END contiguous worker ranges and original-order reduction are unchanged.
    COAL_COMPILER_DIAGNOSTIC_PUSH
    COAL_COMPILER_DIAGNOSTIC_IGNORED_DEPRECECATED_DECLARATIONS
    const bool stateful = request.gjk_initial_guess == coal::GJKInitialGuess::CachedGuess
        || request.enable_cached_gjk_guess;
    COAL_COMPILER_DIAGNOSTIC_POP
    if (stateful)
      throw std::invalid_argument("Independent component batches require requests without cached GJK history");
    actor_ = &bp::extract<const coal::CollisionGeometry&>(actor_owner_)();
    const auto ids = component_ids(object, prepared_->size());
    entries_.resize(ids.size());
    worker_count_ = std::min(static_cast<std::size_t>(workers), entries_.size());
    for (std::size_t index = 0; index < ids.size(); ++index) {
      try {
        entries_[index].target = &prepared_->checked_entry(ids[index]);
      } catch (...) {
        // Missing preparation, like a prepared bad shape, is exposed at its
        // own source position rather than during speculative suffix setup.
        entries_[index].error = std::current_exception();
      }
    }
    if (entries_.empty()) return;
    std::fenv_t caller_environment;
    if (std::fegetenv(&caller_environment) != 0)
      throw std::runtime_error("Cannot read component caller floating-point environment");
    {
      const ReleasedGIL released;
      const auto run = [this, &caller_environment](std::size_t lane) noexcept {
        const std::size_t begin = lane * entries_.size() / worker_count_;
        const std::size_t end = (lane + 1) * entries_.size() / worker_count_;
        try {
          const QueryFloatingEnvironment environment(caller_environment);
          for (std::size_t index = begin; index < end; ++index) {
            auto& entry = entries_[index];
            if (!entry.target || !entry.target->ready) continue;
            try {
              // Raw Coal distance creates a fresh solver for each independent
              // item. The original request and every persistent shape stay read-only.
              const coal::DistanceRequest local_request(request_);
              entry.queried = true;
              coal::distance(actor_, actor_transform_, entry.target->shape,
                  entry.target->transform, local_request, entry.distance);
              if (!std::isfinite(entry.distance.min_distance))
                throw std::runtime_error("Independent scene distance returned nonfinite evidence");
            } catch (...) {
              entry.error = std::current_exception();
            }
          }
        } catch (...) {
          const auto error = std::current_exception();
          for (std::size_t index = begin; index < end; ++index)
            if (entries_[index].target && entries_[index].target->ready) entries_[index].error = error;
        }
      };
      std::vector<std::thread> threads;
      threads.reserve(worker_count_ - 1);
      try {
        for (std::size_t lane = 1; lane < worker_count_; ++lane) threads.emplace_back(run, lane);
        run(0);
      } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
      }
      for (auto& thread : threads) thread.join();
    }
    // Read worker-written counters only after joining; there is no shared
    // counter contention. Include failing Coal calls and speculative suffixes.
    for (const auto& entry : entries_) physical_query_count_ += entry.queried;
  }

  bp::tuple prefix(Py_ssize_t start, double stop_distance) const {
    if (start < 0 || static_cast<std::size_t>(start) > entries_.size())
      throw std::out_of_range("Component prefix start is out of range");
    if (!std::isfinite(stop_distance) || stop_distance < 0.)
      throw std::invalid_argument("stop_distance must be finite and nonnegative");
    Py_ssize_t minimum = -1;
    double minimum_distance = kInfinity;
    const coal::DistanceResult* minimum_result = nullptr;
    for (std::size_t index = static_cast<std::size_t>(start); index < entries_.size(); ++index) {
      const auto& entry = entries_[index];
      if (entry.error) std::rethrow_exception(entry.error);
      entry.target->error.raise();
      const Py_ssize_t relative = static_cast<Py_ssize_t>(index) - start;
      if (entry.distance.min_distance <= stop_distance) {
        bp::object result = minimum_result ? bp::object(*minimum_result) : bp::object();
        return bp::make_tuple(relative, minimum, result, relative, bp::object(entry.distance));
      }
      if (entry.distance.min_distance < minimum_distance) {
        // Completion order cannot select a different equal-distance component.
        minimum = relative;
        minimum_distance = entry.distance.min_distance;
        minimum_result = &entry.distance;
      }
    }
    bp::object result = minimum_result ? bp::object(*minimum_result) : bp::object();
    return bp::make_tuple(static_cast<Py_ssize_t>(entries_.size()) - start,
                         minimum, result, -1, bp::object());
  }

  std::size_t size() const { return entries_.size(); }
  std::size_t worker_count() const { return worker_count_; }
  std::size_t physical_query_count() const { return physical_query_count_; }
  std::size_t slot_nbytes() const { return sizeof(ComponentDistanceEntry); }
  std::size_t storage_nbytes() const {
    // Prepared poses and geometry payloads are separately owned/reserved. This
    // reports fixed native batch values, excluding solver/thread transients.
    return sizeof(*this) + entries_.capacity() * sizeof(ComponentDistanceEntry);
  }

 private:
  std::shared_ptr<PreparedComponents> prepared_;
  bp::object actor_owner_;
  const coal::CollisionGeometry* actor_ = nullptr;
  coal::Transform3s actor_transform_;
  coal::DistanceRequest request_;
  std::vector<ComponentDistanceEntry> entries_;
  std::size_t worker_count_ = 0;
  std::size_t physical_query_count_ = 0;
};

std::shared_ptr<ComponentDistanceBatch> PreparedComponents::scan(const bp::object& actor,
    const coal::Transform3s& actor_transform, const bp::object& ids,
    const coal::DistanceRequest& request, int workers) {
  return std::make_shared<ComponentDistanceBatch>(shared_from_this(), actor,
      actor_transform, ids, request, workers);
}

bp::tuple floating_point_mode_key() {
  const int rounding = std::fegetround();
  if (rounding < 0) throw std::runtime_error("Cannot read floating-point rounding mode");
  unsigned int controls = 0;
#if defined(__SSE__)
  // MXCSR bits 0..5 are sticky exception status. Bits 6..15 contain DAZ,
  // exception masks, rounding and FTZ; these can affect NumPy's multiplication.
  controls = _mm_getcsr() & 0xffc0u;
#endif
  return bp::make_tuple(rounding, controls);
}
// END prepared data never caches distances, search certificates or contact policy.
}  // namespace

BOOST_PYTHON_MODULE(_coal_scene_batch) {
  if (_import_array() < 0) boost::python::throw_error_already_set();
  // The installed module registers CollisionGeometry, Transform3s, request and
  // result converters. Reuse those exact types rather than defining a second
  // binding system or reconstructing geometry in this project.
  boost::python::import("coal");
  boost::python::scope().attr("MAX_BATCH_SIZE") = kMaximumBatchSize;
  boost::python::scope().attr("MAX_PROJECTION_DIRECTIONS") = kMaximumProjectionDirections;
  // BEGIN prepared component bindings own geometry and preserve source-order evidence.
  bp::class_<ComponentTable, std::shared_ptr<ComponentTable>, boost::noncopyable>(
      "ComponentTable", bp::init<Py_ssize_t>((bp::arg("count"))))
      .add_property("size", &ComponentTable::size)
      .add_property("slot_nbytes", &ComponentTable::slot_nbytes)
      .add_property("storage_nbytes", &ComponentTable::storage_nbytes);
  bp::class_<PreparedComponents, std::shared_ptr<PreparedComponents>, boost::noncopyable>(
      "PreparedComponents", bp::init<std::shared_ptr<ComponentTable>>((bp::arg("table"))))
      .def("missing", &PreparedComponents::missing, (bp::arg("ids")))
      .def("register", &PreparedComponents::register_components,
           (bp::arg("ids"), bp::arg("shapes"), bp::arg("world_poses")))
      .def("pose", &PreparedComponents::pose, (bp::arg("flat_id")))
      .def("scan", &PreparedComponents::scan,
           (bp::arg("actor_shape"), bp::arg("actor_transform"), bp::arg("ids"),
            bp::arg("request"), bp::arg("workers") = 8))
      .add_property("size", &PreparedComponents::size)
      .add_property("slot_nbytes", &PreparedComponents::slot_nbytes)
      .add_property("storage_nbytes", &PreparedComponents::storage_nbytes);
  bp::class_<ComponentDistanceBatch, std::shared_ptr<ComponentDistanceBatch>, boost::noncopyable>(
      "ComponentDistanceBatch", bp::no_init)
      .def("prefix", &ComponentDistanceBatch::prefix,
           (bp::arg("start"), bp::arg("stop_distance")),
           "Return (count, minimum_index, minimum_result, near_index, near_result).\n"
           "Indices are relative to start. Only errors through the first near item are exposed.\n"
           "Already computed suffix slots can be consumed by a later prefix call.")
      .def("__len__", &ComponentDistanceBatch::size)
      .add_property("worker_count", &ComponentDistanceBatch::worker_count)
      .add_property("physical_query_count", &ComponentDistanceBatch::physical_query_count)
      .add_property("slot_nbytes", &ComponentDistanceBatch::slot_nbytes)
      .add_property("storage_nbytes", &ComponentDistanceBatch::storage_nbytes);
  bp::def("floating_point_mode_key", &floating_point_mode_key,
          "Return rounding and platform arithmetic-control bits, excluding exception status flags.");
  // END one immutable source table can back successive exact owner-pose instances.
  // BEGIN owned ordered results defer every item failure to its consumption.
  bp::class_<OrderedDistanceBatch, std::shared_ptr<OrderedDistanceBatch>, boost::noncopyable>(
      "OrderedDistanceBatch", bp::no_init)
      .def("result", &OrderedDistanceBatch::result, (bp::arg("index")))
      .def("__len__", &OrderedDistanceBatch::size)
      .add_property("worker_count", &OrderedDistanceBatch::worker_count)
      .add_property("storage_nbytes", &OrderedDistanceBatch::storage_nbytes);
  bp::def("ordered_distances", &ordered_distances,
      (bp::arg("first_shapes"), bp::arg("first_transforms"), bp::arg("second_shapes"),
       bp::arg("second_transforms"), bp::arg("request"), bp::arg("workers") = 8),
      "Evaluate independent cold-request distances in input slots with at most eight workers.\n"
      "Transforms are copied without arithmetic. No request history is propagated between items.\n"
      "batch.result(index) returns a fresh result copy or rethrows that item's deferred exception.\n"
      "Keep the batch or original geometries alive while consuming Coal result geometry pointers.");
  // END ordinary prefix scans and spatial queries retain their original order.
  // BEGIN snapshot-owned tree and ephemeral-query bindings share Coal types.
  bp::class_<ComponentAABBTree, std::shared_ptr<ComponentAABBTree>, boost::noncopyable>(
      "ComponentAABBTree", bp::init<bp::object, bp::object, bp::object>(
          (bp::arg("lower"), bp::arg("upper"), bp::arg("prunable"))))
      .add_property("size", &ComponentAABBTree::size)
      .add_property("storage_nbytes", &ComponentAABBTree::storage_nbytes)
      .def("overlap_ids", &ComponentAABBTree::overlap_ids,
           (bp::arg("lower"), bp::arg("upper")))
      .def("query", &ComponentAABBTree::query,
           (bp::arg("actor_shape"), bp::arg("actor_transform"), bp::arg("owner_transform"),
            bp::arg("candidate_ids"), bp::arg("stop_distance")));
  bp::class_<SpatialQuery, std::shared_ptr<SpatialQuery>, boost::noncopyable>("SpatialQuery", bp::no_init)
      .add_property("workspace_nbytes", &SpatialQuery::workspace_nbytes)
      .def("seed_id", &SpatialQuery::seed_id, (bp::arg("prior_id") = -1))
      .def("retained_ids", &SpatialQuery::retained_ids, (bp::arg("upper_bound")));
  bp::def("certified_distance_upper_bound", &certified_distance_upper_bound,
      (bp::arg("actor_shape"), bp::arg("actor_transform"), bp::arg("target_shape"),
       bp::arg("target_transform"), bp::arg("distance_result")),
      "Return an outward upper distance from feasible native-geometry points; infinity if uncertified.");
  // BEGIN raw TSDF traversal consumes the same maintained native constructions.
  bp::def("native_projection_bounds", &native_projection_bounds,
      (bp::arg("geometry"), bp::arg("transform"), bp::arg("direction")),
      "Return an outward projection interval for the full registered native shape.");
  bp::def("native_feasible_point_bounds", &native_feasible_point_bounds,
      (bp::arg("geometry"), bp::arg("transform"), bp::arg("wanted_world")),
      "Return local and world enclosures of the same actual native-shape point.");
  // END no mesh is constructed from a task TSDF.
  // END no contact rule or prior-state result is stored in the native index.
}
// END near-contact decisions and result consumption stay in Python's original order.
