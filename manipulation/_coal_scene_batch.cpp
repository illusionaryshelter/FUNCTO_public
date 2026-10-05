// Ordered leaf distances and conservative spatial search using installed Coal.
// Static trees own only component bounds/IDs. Query objects contain current
// support certificates, never contact policy, solver history or old distances.

// BEGIN exact Coal distance scans stop before any later near-contact work.
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
#include <coal/shape/geometric_shapes.h>
#include <Eigen/QR>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

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
  const RoundingGuard rounding;
  first = transformed_point(first, actor_transform);
  second = transformed_point(second, target_transform);
  IntervalPoint difference;
  for (int axis = 0; axis < 3; ++axis) difference[axis] = first[axis] - second[axis];
  const double upper = norm_interval(difference).upper();
  return std::isfinite(upper) && upper >= 0. ? upper : kInfinity;
}
// END certificates affect search only; original Coal results remain untouched.

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
  boost::python::scope().attr("MAX_PROJECTION_DIRECTIONS") = kMaximumProjectionDirections;
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
  // END no contact rule or prior-state result is stored in the native index.
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
