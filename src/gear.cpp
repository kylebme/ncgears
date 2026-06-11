#include "ncgear/gear.hpp"

#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/intersections.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ncgear {
namespace {

using Complex = std::complex<double>;
using Segment = Kernel::Segment_2;
using ScalarFunction = std::function<double(double)>;
using CurveFunction = std::function<Complex(double)>;

enum class Flank : int {
  kMinus = -1,
  kPlus = 1,
};

constexpr std::array<double, 8> kGaussNodes = {
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
};

constexpr std::array<double, 8> kGaussWeights = {
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
};

double flank_sign(Flank flank) {
  return static_cast<double>(static_cast<int>(flank));
}

Complex cis(double angle) {
  return {std::cos(angle), std::sin(angle)};
}

Point to_point(const Complex& value) {
  return {value.real(), value.imag()};
}

double point_distance(const Point& lhs, const Point& rhs) {
  const double dx = lhs.x() - rhs.x();
  const double dy = lhs.y() - rhs.y();
  return std::hypot(dx, dy);
}

double project_fraction(const Point& start, const Point& end, const Point& point) {
  const double dx = end.x() - start.x();
  const double dy = end.y() - start.y();
  const double denominator = dx * dx + dy * dy;
  if (denominator <= 0.0) {
    return 0.0;
  }
  const double numerator =
      (point.x() - start.x()) * dx + (point.y() - start.y()) * dy;
  return std::clamp(numerator / denominator, 0.0, 1.0);
}

double oriented_sign(double value, double fallback) {
  if (std::abs(value) < 1e-12) {
    return fallback;
  }
  return value < 0.0 ? -1.0 : 1.0;
}

std::string json_escape(const std::string& value) {
  std::ostringstream output;
  for (const char character : value) {
    if (character == '"' || character == '\\') {
      output << '\\';
    }
    output << character;
  }
  return output.str();
}

class IntegralTable {
 public:
  IntegralTable(
      ScalarFunction function,
      double domain_start,
      double domain_end,
      bool periodic,
      int interval_count = 1 << 14)
      : function_(std::move(function)),
        domain_start_(domain_start),
        domain_end_(domain_end),
        periodic_(periodic),
        step_((domain_end - domain_start) / static_cast<double>(interval_count)),
        prefix_(static_cast<std::size_t>(interval_count + 1), 0.0) {
    if (!(domain_start_ < domain_end_)) {
      throw std::invalid_argument("Integral domain must have positive length.");
    }
    for (int index = 0; index < interval_count; ++index) {
      const double start = domain_start_ + static_cast<double>(index) * step_;
      prefix_[static_cast<std::size_t>(index + 1)] =
          prefix_[static_cast<std::size_t>(index)] + integrate_small(start, start + step_);
    }
    domain_integral_ = prefix_.back();
  }

  double domain_integral() const {
    return domain_integral_;
  }

  double integral(double start, double end) const {
    return antiderivative(end) - antiderivative(start);
  }

 private:
  double integrate_small(double start, double end) const {
    const double midpoint = 0.5 * (start + end);
    const double radius = 0.5 * (end - start);
    double sum = 0.0;
    for (std::size_t index = 0; index < kGaussNodes.size(); ++index) {
      sum +=
          kGaussWeights[index] * function_(midpoint + radius * kGaussNodes[index]);
    }
    return radius * sum;
  }

  double antiderivative(double value) const {
    double cycles = 0.0;
    double wrapped = value;
    const double domain_length = domain_end_ - domain_start_;
    if (periodic_) {
      cycles = std::floor((value - domain_start_) / domain_length);
      wrapped = value - cycles * domain_length;
      if (wrapped < domain_start_) {
        wrapped += domain_length;
      } else if (wrapped >= domain_end_) {
        wrapped -= domain_length;
      }
    } else {
      if (value < domain_start_ - 1e-10 || value > domain_end_ + 1e-10) {
        throw std::out_of_range("Integral query is outside the sampled transmission domain.");
      }
      wrapped = std::clamp(value, domain_start_, domain_end_);
    }
    const int interval = std::clamp(
        static_cast<int>(std::floor((wrapped - domain_start_) / step_)),
        0,
        static_cast<int>(prefix_.size()) - 2);
    const double interval_start = domain_start_ + static_cast<double>(interval) * step_;
    return cycles * domain_integral_ + prefix_[static_cast<std::size_t>(interval)] +
           integrate_small(interval_start, wrapped);
  }

  ScalarFunction function_;
  double domain_start_ = 0.0;
  double domain_end_ = 0.0;
  bool periodic_ = false;
  double step_ = 0.0;
  double domain_integral_ = 0.0;
  std::vector<double> prefix_;
};

struct CurveIntersection {
  double lhs = 0.0;
  double rhs = 0.0;
  Complex point{0.0, 0.0};
  double residual = std::numeric_limits<double>::infinity();
};

struct DriveGeometry {
  double minus_fillet_transition = 0.0;
  double minus_flank_transition = 0.0;
  double minus_fillet_dedendum = 0.0;
  double minus_flank_addendum = 0.0;
  double minus_addendum = 0.0;
  double plus_addendum = 0.0;
  double plus_flank_addendum = 0.0;
  double plus_flank_transition = 0.0;
  double plus_fillet_transition = 0.0;
  double plus_fillet_dedendum = 0.0;
};

struct DrivenGeometry {
  double minus_flank_addendum = 0.0;
  double minus_addendum = 0.0;
  double minus_flank_transition = 0.0;
  double minus_fillet_transition = 0.0;
  double minus_fillet_dedendum = 0.0;
  double plus_fillet_dedendum = 0.0;
  double plus_fillet_transition = 0.0;
  double plus_flank_transition = 0.0;
  double plus_flank_addendum = 0.0;
  double plus_addendum = 0.0;
};

struct SampledPoint {
  double parameter = 0.0;
  Point point{0.0, 0.0};
};

std::vector<double> find_roots(
    const ScalarFunction& function,
    double low,
    double high,
    int samples) {
  std::vector<double> roots;
  double previous_x = low;
  double previous_value = function(previous_x);
  const double step = (high - low) / static_cast<double>(samples);

  auto add_root = [&roots](double root) {
    if (roots.empty() || std::abs(roots.back() - root) > 1e-8) {
      roots.push_back(root);
    }
  };

  for (int index = 1; index <= samples; ++index) {
    const double current_x =
        index == samples ? high : low + static_cast<double>(index) * step;
    const double current_value = function(current_x);
    if (!std::isfinite(previous_value) || !std::isfinite(current_value)) {
      previous_x = current_x;
      previous_value = current_value;
      continue;
    }
    if (std::abs(previous_value) < 1e-12) {
      add_root(previous_x);
    }
    if ((previous_value < 0.0 && current_value > 0.0) ||
        (previous_value > 0.0 && current_value < 0.0)) {
      double left = previous_x;
      double right = current_x;
      double left_value = previous_value;
      for (int iteration = 0; iteration < 90; ++iteration) {
        const double midpoint = 0.5 * (left + right);
        const double midpoint_value = function(midpoint);
        if (std::abs(midpoint_value) < 1e-14 || right - left < 1e-13) {
          left = midpoint;
          right = midpoint;
          break;
        }
        if ((left_value < 0.0 && midpoint_value > 0.0) ||
            (left_value > 0.0 && midpoint_value < 0.0)) {
          right = midpoint;
        } else {
          left = midpoint;
          left_value = midpoint_value;
        }
      }
      add_root(0.5 * (left + right));
    }
    previous_x = current_x;
    previous_value = current_value;
  }
  if (std::abs(previous_value) < 1e-12) {
    add_root(high);
  }
  return roots;
}

double choose_directional_root(
    const ScalarFunction& function,
    double reference,
    int direction,
    double span) {
  for (const double multiplier : {1.0, 1.5, 2.0}) {
    const double low = reference - span * multiplier;
    const double high = reference + span * multiplier;
    const std::vector<double> roots = find_roots(function, low, high, 4096);
    std::optional<double> best;
    for (const double root : roots) {
      const bool correct_side =
          direction < 0 ? root < reference - 1e-8 : root > reference + 1e-8;
      if (!correct_side) {
        continue;
      }
      if (!best || std::abs(root - reference) < std::abs(*best - reference)) {
        best = root;
      }
    }
    if (best) {
      return *best;
    }
  }
  throw std::runtime_error("Unable to bracket a required directional root.");
}

CurveIntersection refine_intersection(
    const CurveFunction& lhs,
    double lhs_low,
    double lhs_high,
    const CurveFunction& rhs,
    double rhs_low,
    double rhs_high,
    double lhs_guess,
    double rhs_guess) {
  double lhs_parameter = std::clamp(lhs_guess, lhs_low, lhs_high);
  double rhs_parameter = std::clamp(rhs_guess, rhs_low, rhs_high);

  for (int iteration = 0; iteration < 60; ++iteration) {
    const Complex difference = lhs(lhs_parameter) - rhs(rhs_parameter);
    const double residual = std::abs(difference);
    if (residual < 1e-11) {
      return {
          lhs_parameter,
          rhs_parameter,
          0.5 * (lhs(lhs_parameter) + rhs(rhs_parameter)),
          residual,
      };
    }

    const double lhs_step = std::max(1e-7, 1e-6 * std::abs(lhs_high - lhs_low));
    const double rhs_step = std::max(1e-7, 1e-6 * std::abs(rhs_high - rhs_low));
    const Complex lhs_derivative =
        (lhs(lhs_parameter + lhs_step) - lhs(lhs_parameter - lhs_step)) /
        (2.0 * lhs_step);
    const Complex rhs_derivative =
        (rhs(rhs_parameter + rhs_step) - rhs(rhs_parameter - rhs_step)) /
        (2.0 * rhs_step);

    const double a = lhs_derivative.real();
    const double b = -rhs_derivative.real();
    const double c = lhs_derivative.imag();
    const double d = -rhs_derivative.imag();
    const double determinant = a * d - b * c;
    if (std::abs(determinant) < 1e-14) {
      break;
    }

    const double delta_lhs =
        (-difference.real() * d + b * difference.imag()) / determinant;
    const double delta_rhs =
        (-a * difference.imag() + difference.real() * c) / determinant;

    double damping = 1.0;
    bool accepted = false;
    while (damping >= 1.0 / 1024.0) {
      const double candidate_lhs =
          std::clamp(lhs_parameter + damping * delta_lhs, lhs_low, lhs_high);
      const double candidate_rhs =
          std::clamp(rhs_parameter + damping * delta_rhs, rhs_low, rhs_high);
      if (std::abs(lhs(candidate_lhs) - rhs(candidate_rhs)) < residual) {
        lhs_parameter = candidate_lhs;
        rhs_parameter = candidate_rhs;
        accepted = true;
        break;
      }
      damping *= 0.5;
    }
    if (!accepted) {
      break;
    }
  }

  const Complex lhs_point = lhs(lhs_parameter);
  const Complex rhs_point = rhs(rhs_parameter);
  return {
      lhs_parameter,
      rhs_parameter,
      0.5 * (lhs_point + rhs_point),
      std::abs(lhs_point - rhs_point),
  };
}

std::vector<CurveIntersection> find_curve_intersections(
    const CurveFunction& lhs,
    double lhs_low,
    double lhs_high,
    const CurveFunction& rhs,
    double rhs_low,
    double rhs_high,
    int sample_count = 320) {
  std::vector<SampledPoint> lhs_points;
  std::vector<SampledPoint> rhs_points;
  lhs_points.reserve(static_cast<std::size_t>(sample_count + 1));
  rhs_points.reserve(static_cast<std::size_t>(sample_count + 1));

  for (int index = 0; index <= sample_count; ++index) {
    const double fraction = static_cast<double>(index) / static_cast<double>(sample_count);
    const double lhs_parameter = std::lerp(lhs_low, lhs_high, fraction);
    const double rhs_parameter = std::lerp(rhs_low, rhs_high, fraction);
    lhs_points.push_back({lhs_parameter, to_point(lhs(lhs_parameter))});
    rhs_points.push_back({rhs_parameter, to_point(rhs(rhs_parameter))});
  }

  std::vector<CurveIntersection> intersections;
  for (int lhs_index = 0; lhs_index < sample_count; ++lhs_index) {
    const Segment lhs_segment(
        lhs_points[static_cast<std::size_t>(lhs_index)].point,
        lhs_points[static_cast<std::size_t>(lhs_index + 1)].point);
    for (int rhs_index = 0; rhs_index < sample_count; ++rhs_index) {
      const Segment rhs_segment(
          rhs_points[static_cast<std::size_t>(rhs_index)].point,
          rhs_points[static_cast<std::size_t>(rhs_index + 1)].point);
      if (!CGAL::do_intersect(lhs_segment, rhs_segment)) {
        continue;
      }
      const auto result = CGAL::intersection(lhs_segment, rhs_segment);
      if (!result) {
        continue;
      }

      Point intersection_point;
      if (const Point* point = std::get_if<Point>(&*result)) {
        intersection_point = *point;
      } else if (const Segment* overlap = std::get_if<Segment>(&*result)) {
        intersection_point = CGAL::midpoint(overlap->source(), overlap->target());
      } else {
        continue;
      }

      const auto& lhs_start = lhs_points[static_cast<std::size_t>(lhs_index)];
      const auto& lhs_end = lhs_points[static_cast<std::size_t>(lhs_index + 1)];
      const auto& rhs_start = rhs_points[static_cast<std::size_t>(rhs_index)];
      const auto& rhs_end = rhs_points[static_cast<std::size_t>(rhs_index + 1)];
      const double lhs_fraction =
          project_fraction(lhs_start.point, lhs_end.point, intersection_point);
      const double rhs_fraction =
          project_fraction(rhs_start.point, rhs_end.point, intersection_point);
      const double lhs_guess = std::lerp(lhs_start.parameter, lhs_end.parameter, lhs_fraction);
      const double rhs_guess = std::lerp(rhs_start.parameter, rhs_end.parameter, rhs_fraction);
      CurveIntersection refined = refine_intersection(
          lhs,
          lhs_low,
          lhs_high,
          rhs,
          rhs_low,
          rhs_high,
          lhs_guess,
          rhs_guess);
      if (refined.residual > 1e-7) {
        continue;
      }

      const bool duplicate = std::any_of(
          intersections.begin(),
          intersections.end(),
          [&refined](const CurveIntersection& existing) {
            return std::abs(existing.lhs - refined.lhs) < 1e-5 &&
                   std::abs(existing.rhs - refined.rhs) < 1e-5;
          });
      if (!duplicate) {
        intersections.push_back(refined);
      }
    }
  }
  return intersections;
}

template <typename Predicate, typename Score>
CurveIntersection choose_intersection(
    const std::vector<CurveIntersection>& intersections,
    Predicate predicate,
    Score score,
    const std::string& label) {
  const CurveIntersection* best = nullptr;
  double best_score = std::numeric_limits<double>::infinity();
  for (const auto& intersection : intersections) {
    if (!predicate(intersection)) {
      continue;
    }
    const double candidate_score = score(intersection);
    if (candidate_score < best_score) {
      best = &intersection;
      best_score = candidate_score;
    }
  }
  if (best == nullptr) {
    throw std::runtime_error("No valid CGAL intersection found for " + label + ".");
  }
  return *best;
}

std::string self_intersection_description(const std::vector<Point>& points) {
  if (points.size() < 4) {
    return "too few points";
  }
  const std::size_t segment_count = points.size() - 1;
  for (std::size_t lhs = 0; lhs < segment_count; ++lhs) {
    const Segment lhs_segment(points[lhs], points[lhs + 1]);
    for (std::size_t rhs = lhs + 1; rhs < segment_count; ++rhs) {
      const bool adjacent = rhs == lhs + 1 || (lhs == 0 && rhs + 1 == segment_count);
      if (adjacent) {
        continue;
      }
      const Segment rhs_segment(points[rhs], points[rhs + 1]);
      if (CGAL::do_intersect(lhs_segment, rhs_segment)) {
        std::ostringstream message;
        message << "segments " << lhs << " and " << rhs;
        return message.str();
      }
    }
  }
  return "CGAL Polygon_2 rejected the outline without a nonadjacent segment crossing";
}

double placed_pair_overlap_area(
    const std::vector<Point>& drive,
    const std::vector<Point>& driven,
    double center_distance,
    double drive_angle,
    double driven_angle) {
  using ExactKernel = CGAL::Exact_predicates_exact_constructions_kernel;
  using ExactPoint = ExactKernel::Point_2;
  using ExactPolygon = CGAL::Polygon_2<ExactKernel>;
  using ExactPolygonSet = CGAL::Polygon_set_2<ExactKernel>;
  using ExactPolygonWithHoles = CGAL::Polygon_with_holes_2<ExactKernel>;

  auto make_polygon = [](
                          const std::vector<Point>& points,
                          double angle,
                          double translate_x) {
    ExactPolygon polygon;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    for (std::size_t index = 0; index + 1 < points.size(); ++index) {
      const double x =
          cosine * points[index].x() - sine * points[index].y() + translate_x;
      const double y =
          sine * points[index].x() + cosine * points[index].y();
      polygon.push_back(ExactPoint(x, y));
    }
    if (polygon.orientation() == CGAL::CLOCKWISE) {
      polygon.reverse_orientation();
    }
    return polygon;
  };

  const ExactPolygon drive_polygon = make_polygon(drive, drive_angle, 0.0);
  const ExactPolygon driven_polygon =
      make_polygon(driven, driven_angle, center_distance);
  ExactPolygonSet overlap;
  overlap.insert(drive_polygon);
  overlap.intersection(driven_polygon);

  std::list<ExactPolygonWithHoles> components;
  overlap.polygons_with_holes(std::back_inserter(components));
  ExactKernel::FT area = 0;
  for (const ExactPolygonWithHoles& component : components) {
    area += CGAL::abs(component.outer_boundary().area());
    for (auto hole = component.holes_begin(); hole != component.holes_end(); ++hole) {
      area -= CGAL::abs(hole->area());
    }
  }
  return CGAL::to_double(area);
}

}  // namespace

struct GearGenerator::Impl {
  explicit Impl(SampleConfig sample)
      : config(std::move(sample)),
        alpha(config.pressure_angle_deg * kPi / 180.0),
        addendum(config.addendum_factor * config.module),
        dedendum(config.dedendum_factor * config.module),
        fillet_radius(config.fillet_factor * config.module),
        integral(
            [this](double phi) { return integral_density(phi); },
            domain_start(),
            domain_end(),
            is_closed()) {
    validate_config();
    drive_teeth = config.teeth;
    active_start = is_closed() ? domain_start() : config.transmission.active_start;
    active_end = is_closed() ? domain_start() + period() : config.transmission.active_end;
    total_integral = arc_integral(active_start, active_end);
    center_distance =
        static_cast<double>(drive_teeth) * kPi * config.module / total_integral;
    mean_pitch_phi = (active_end - active_start) / static_cast<double>(drive_teeth);
    if (is_closed()) {
      const double exact_driven_teeth =
          static_cast<double>(drive_teeth) * period() / cycle_delta();
      driven_teeth = static_cast<int>(std::lround(exact_driven_teeth));
      if (driven_teeth <= 0 ||
          std::abs(exact_driven_teeth - static_cast<double>(driven_teeth)) > 1e-7) {
        throw std::invalid_argument(
            "Closed transmission requires z1 * period / cycle_delta to be an integer.");
      }
      drive_cycle = period();
      driven_cycle = period() * static_cast<double>(driven_teeth) /
                     static_cast<double>(drive_teeth);
    } else {
      driven_teeth = drive_teeth;
      drive_cycle = active_end - active_start;
      driven_cycle = drive_cycle;
    }
    average_angular_ratio =
        (psi(active_end) - psi(active_start)) / (active_end - active_start);
    curvature_limit =
        std::pow(std::sin(alpha), 2) /
        (dedendum - fillet_radius * (1.0 - std::sin(alpha)));
  }

  bool has_sampled_transmission() const {
    return !config.transmission.psi.empty();
  }

  bool is_closed() const {
    return config.topology == GearTopology::kClosed;
  }

  double domain_start() const {
    return has_sampled_transmission() ? config.transmission.domain_start : 0.0;
  }

  double domain_end() const {
    return has_sampled_transmission() ? config.transmission.domain_end : 2.0 * kPi;
  }

  double period() const {
    return has_sampled_transmission() ? config.transmission.period : 2.0 * kPi;
  }

  double cycle_delta() const {
    return has_sampled_transmission() ? config.transmission.cycle_delta : 2.0 * kPi;
  }

  void validate_config() const {
    if (config.name.empty() || config.teeth < 6 || config.module <= 0.0) {
      throw std::invalid_argument("Invalid sample name, tooth count, or module.");
    }
    if (!(alpha > 0.0 && alpha < 0.5 * kPi)) {
      throw std::invalid_argument("Pressure angle must be between 0 and 90 degrees.");
    }
    if (!(addendum > 0.0 && dedendum > fillet_radius && fillet_radius > 0.0)) {
      throw std::invalid_argument("Invalid rack-cutter dimensions.");
    }
    if (has_sampled_transmission()) {
      const std::size_t sample_count = config.transmission.psi.size();
      if (sample_count < 256 ||
          config.transmission.psi1.size() != sample_count ||
          config.transmission.psi2.size() != sample_count ||
          config.transmission.psi3.size() != sample_count) {
        throw std::invalid_argument(
            "Sampled transmission arrays must have equal length of at least 256.");
      }
      if (!(domain_start() < domain_end())) {
        throw std::invalid_argument("Sampled transmission domain is invalid.");
      }
      if (is_closed()) {
        if (!(period() > 0.0 && cycle_delta() > 0.0)) {
          throw std::invalid_argument("Closed transmission period and cycle delta must be positive.");
        }
      } else if (!(domain_start() < config.transmission.active_start &&
                   config.transmission.active_start < config.transmission.active_end &&
                   config.transmission.active_end < domain_end())) {
        throw std::invalid_argument(
            "Open transmission requires a padded domain around the active interval.");
      }
    } else if (!is_closed()) {
      throw std::invalid_argument("Open gears require sampled transmission data.");
    }
    for (int index = 0; index <= 8192; ++index) {
      const double phi =
          std::lerp(domain_start(), domain_end(), static_cast<double>(index) / 8192.0);
      if (psi1(phi) <= 1e-8) {
        throw std::invalid_argument("Transmission derivative is not strictly positive.");
      }
    }
  }

  double sample_values(
      const std::vector<double>& values,
      double phi,
      double periodic_delta) const {
    const double start = domain_start();
    const double end = domain_end();
    const double length = end - start;
    double cycles = 0.0;
    double x = phi;
    if (is_closed()) {
      cycles = std::floor((phi - start) / period());
      x = phi - cycles * period();
      if (x < start) {
        x += period();
      } else if (x >= start + period()) {
        x -= period();
      }
      const double scaled =
          (x - start) * static_cast<double>(values.size()) / period();
      int index = static_cast<int>(std::floor(scaled));
      index = std::clamp(index, 0, static_cast<int>(values.size()) - 1);
      const int next = (index + 1) % static_cast<int>(values.size());
      const double local = scaled - static_cast<double>(index);
      double right = values[static_cast<std::size_t>(next)];
      if (next == 0) {
        right += periodic_delta;
      }
      return cycles * periodic_delta +
             std::lerp(values[static_cast<std::size_t>(index)], right, local);
    }

    const double clamped = std::clamp(x, start, end);
    const double scaled =
        (clamped - start) * static_cast<double>(values.size() - 1) / length;
    int index = static_cast<int>(std::floor(scaled));
    index = std::clamp(index, 0, static_cast<int>(values.size()) - 2);
    const double local = scaled - static_cast<double>(index);
    return std::lerp(
        values[static_cast<std::size_t>(index)],
        values[static_cast<std::size_t>(index + 1)],
        local);
  }

  double psi(double phi) const {
    if (has_sampled_transmission()) {
      return sample_values(config.transmission.psi, phi, cycle_delta());
    }
    double value = phi;
    for (const Harmonic& harmonic : config.harmonics) {
      value += harmonic.amplitude *
               std::sin(static_cast<double>(harmonic.order) * phi);
    }
    return value;
  }

  double psi1(double phi) const {
    if (has_sampled_transmission()) {
      return sample_values(config.transmission.psi1, phi, 0.0);
    }
    double value = 1.0;
    for (const Harmonic& harmonic : config.harmonics) {
      const double order = static_cast<double>(harmonic.order);
      value += harmonic.amplitude * order * std::cos(order * phi);
    }
    return value;
  }

  double psi2(double phi) const {
    if (has_sampled_transmission()) {
      return sample_values(config.transmission.psi2, phi, 0.0);
    }
    double value = 0.0;
    for (const Harmonic& harmonic : config.harmonics) {
      const double order = static_cast<double>(harmonic.order);
      value -= harmonic.amplitude * order * order * std::sin(order * phi);
    }
    return value;
  }

  double psi3(double phi) const {
    if (has_sampled_transmission()) {
      return sample_values(config.transmission.psi3, phi, 0.0);
    }
    double value = 0.0;
    for (const Harmonic& harmonic : config.harmonics) {
      const double order = static_cast<double>(harmonic.order);
      value -=
          harmonic.amplitude * order * order * order * std::cos(order * phi);
    }
    return value;
  }

  double w(double phi) const {
    const double first = psi1(phi);
    const double second = psi2(phi);
    return std::hypot(second, first * (1.0 + first));
  }

  double integral_density(double phi) const {
    const double first = psi1(phi);
    return w(phi) / std::pow(1.0 + first, 2);
  }

  double arc_integral(double start, double end) const {
    return integral.integral(start, end);
  }

  double drive_radius(double phi) const {
    const double first = psi1(phi);
    return center_distance * first / (1.0 + first);
  }

  double driven_radius(double phi) const {
    return center_distance / (1.0 + psi1(phi));
  }

  Complex drive_centrode(double phi) const {
    return drive_radius(phi) * cis(-phi);
  }

  Complex driven_centrode(double phi) const {
    return -driven_radius(phi) * cis(psi(phi));
  }

  Complex drive_tangent(double phi) const {
    const double first = psi1(phi);
    return Complex(psi2(phi), -first * (1.0 + first)) / w(phi) * cis(-phi);
  }

  Complex driven_tangent(double phi) const {
    const double first = psi1(phi);
    return Complex(psi2(phi), -first * (1.0 + first)) / w(phi) * cis(psi(phi));
  }

  double drive_h(double phi) const {
    const double first = psi1(phi);
    const double second = psi2(phi);
    const double numerator =
        (1.0 + first) *
        (first * (psi3(phi) - first - first * first) - 2.0 * second * second);
    return numerator / std::pow(w(phi), 2);
  }

  double driven_h(double phi) const {
    const double first = psi1(phi);
    const double second = psi2(phi);
    const double numerator =
        (1.0 + first) *
        (first * (psi3(phi) + first * first + first * first * first) -
         second * second);
    return numerator / std::pow(w(phi), 2);
  }

  double drive_kappa(double phi) const {
    const double first = psi1(phi);
    return std::pow(1.0 + first, 2) * drive_h(phi) /
           (center_distance * w(phi));
  }

  double driven_kappa(double phi) const {
    const double first = psi1(phi);
    return std::pow(1.0 + first, 2) * driven_h(phi) /
           (center_distance * w(phi));
  }

  double lambda(int tooth, Flank flank, double phi) const {
    return flank_sign(flank) * kPi * config.module / 4.0 -
           center_distance * arc_integral(chis[static_cast<std::size_t>(tooth)], phi);
  }

  Complex drive_flank(int tooth, Flank flank, double phi) const {
    const double sign = flank_sign(flank);
    return drive_centrode(phi) +
           lambda(tooth, flank, phi) * drive_tangent(phi) * cis(sign * alpha) *
               std::cos(alpha);
  }

  Complex driven_flank(int tooth, Flank flank, double phi) const {
    const double sign = flank_sign(flank);
    return driven_centrode(phi) +
           lambda(tooth, flank, phi) * driven_tangent(phi) * cis(sign * alpha) *
               std::cos(alpha);
  }

  Complex drive_addendum(double phi) const {
    return drive_centrode(phi) + addendum * Complex(0.0, 1.0) * drive_tangent(phi);
  }

  Complex drive_dedendum(double phi) const {
    return drive_centrode(phi) - dedendum * Complex(0.0, 1.0) * drive_tangent(phi);
  }

  Complex driven_addendum(double phi) const {
    return driven_centrode(phi) - addendum * Complex(0.0, 1.0) * driven_tangent(phi);
  }

  Complex driven_dedendum(double phi) const {
    return driven_centrode(phi) + dedendum * Complex(0.0, 1.0) * driven_tangent(phi);
  }

  Complex drive_fillet(int tooth, Flank flank, double phi) const {
    const double sign = flank_sign(flank);
    const double root_height = dedendum - fillet_radius;
    const Complex offset(
        lambda(tooth, flank, phi) + sign * fillet_radius / std::cos(alpha) +
            sign * root_height * std::tan(alpha),
        -root_height);
    const Complex midpoint = drive_centrode(phi) + offset * drive_tangent(phi);
    const double orientation = oriented_sign(drive_h(phi), -1.0);
    const Complex normal =
        -orientation * offset / std::abs(offset) * drive_tangent(phi);
    return midpoint + fillet_radius * normal;
  }

  Complex driven_fillet(int tooth, Flank flank, double phi) const {
    const double sign = flank_sign(flank);
    const double root_height = dedendum - fillet_radius;
    const Complex offset(
        lambda(tooth, flank, phi) - sign * fillet_radius / std::cos(alpha) -
            sign * root_height * std::tan(alpha),
        root_height);
    const Complex midpoint = driven_centrode(phi) + offset * driven_tangent(phi);
    const double orientation = oriented_sign(driven_h(phi), 1.0);
    const Complex normal =
        orientation * offset / std::abs(offset) * driven_tangent(phi);
    return midpoint + fillet_radius * normal;
  }

  void validate_centrodes() const {
    double maximum_drive_kappa = -std::numeric_limits<double>::infinity();
    double minimum_driven_kappa = std::numeric_limits<double>::infinity();
    std::vector<Point> drive;
    std::vector<Point> driven;
    drive.reserve(4096);
    driven.reserve(4096);
    for (int index = 0; index < 4096; ++index) {
      const double fraction = static_cast<double>(index) / 4096.0;
      const double drive_phi =
          is_closed() ? active_start + fraction * drive_cycle
                      : std::lerp(domain_start(), domain_end(), fraction);
      const double driven_phi =
          is_closed() ? active_start + fraction * driven_cycle : drive_phi;
      maximum_drive_kappa =
          std::max(maximum_drive_kappa, drive_kappa(drive_phi));
      minimum_driven_kappa =
          std::min(minimum_driven_kappa, driven_kappa(driven_phi));
      if (is_closed()) {
        drive.push_back(to_point(drive_centrode(drive_phi)));
        driven.push_back(to_point(driven_centrode(driven_phi)));
      }
    }
    if (maximum_drive_kappa > 1e-9 || minimum_driven_kappa < -1e-9) {
      throw std::runtime_error("Sample centrodes are not convex with paper-compatible orientation.");
    }
    if (is_closed()) {
      CGAL::Polygon_2<Kernel> drive_polygon(drive.begin(), drive.end());
      CGAL::Polygon_2<Kernel> driven_polygon(driven.begin(), driven.end());
      if (!drive_polygon.is_simple() || !driven_polygon.is_simple()) {
        throw std::runtime_error("Sample centrode self-intersects.");
      }
    }
  }

  void solve_chis() {
    chis.clear();
    const int count =
        is_closed() ? std::max(drive_teeth, driven_teeth)
                    : drive_teeth + 2 * padding_teeth;
    chis.reserve(static_cast<std::size_t>(count));
    for (int tooth = 0; tooth < count; ++tooth) {
      const int logical_tooth = is_closed() ? tooth : tooth - padding_teeth;
      const double offset =
          is_closed() ? static_cast<double>(logical_tooth)
                      : static_cast<double>(logical_tooth) + 0.5;
      const double target = offset * kPi * config.module;
      const ScalarFunction equation = [this, target](double phi) {
        return center_distance * arc_integral(active_start, phi) - target;
      };
      const double low = is_closed() ? active_start : domain_start();
      const double high =
          is_closed() ? active_start + std::max(drive_cycle, driven_cycle) : domain_end();
      const std::vector<double> roots = find_roots(equation, low, high, 8192);
      if (roots.empty()) {
        throw std::runtime_error("Unable to solve chi(k).");
      }
      chis.push_back(roots.front());
    }
  }

  double directional_root(
      const ScalarFunction& equation,
      double reference,
      int direction,
      double span) const {
    if (is_closed()) {
      return choose_directional_root(equation, reference, direction, span);
    }
    for (const double multiplier : {1.0, 1.5, 2.0}) {
      const double low = std::max(domain_start(), reference - span * multiplier);
      const double high = std::min(domain_end(), reference + span * multiplier);
      const std::vector<double> roots = find_roots(equation, low, high, 4096);
      std::optional<double> best;
      for (const double root : roots) {
        const bool correct_side =
            direction < 0 ? root < reference - 1e-8 : root > reference + 1e-8;
        if (correct_side &&
            (!best || std::abs(root - reference) < std::abs(*best - reference))) {
          best = root;
        }
      }
      if (best) {
        return *best;
      }
    }
    throw std::runtime_error("Unable to bracket a required open-gear root.");
  }

  double singular(int tooth, Flank flank, bool driven) const {
    const double sign = flank_sign(flank);
    const ScalarFunction equation = [this, tooth, flank, driven, sign](double phi) {
      const double curvature = driven ? driven_kappa(phi) : drive_kappa(phi);
      return lambda(tooth, flank, phi) * curvature - sign * std::tan(alpha);
    };
    int direction = 0;
    if (!driven) {
      direction = flank == Flank::kMinus ? -1 : 1;
    } else {
      direction = flank == Flank::kMinus ? 1 : -1;
    }
    return directional_root(
        equation,
        chis[static_cast<std::size_t>(tooth)],
        direction,
        5.0 * mean_pitch_phi);
  }

  double solve_lambda_target(
      int tooth,
      Flank flank,
      double target,
      int direction) const {
    const ScalarFunction equation =
        [this, tooth, flank, target](double phi) {
          return lambda(tooth, flank, phi) - target;
        };
    return directional_root(
        equation,
        chis[static_cast<std::size_t>(tooth)],
        direction,
        5.0 * mean_pitch_phi);
  }

  CurveIntersection solve_transition(
      int tooth,
      Flank flank,
      bool driven,
      double contact_parameter,
      double dedendum_parameter) {
    const double chi = chis[static_cast<std::size_t>(tooth)];
    const double previous =
        driven ? previous_driven_chi(tooth) : previous_drive_chi(tooth);
    const double next = driven ? next_driven_chi(tooth) : next_drive_chi(tooth);
    double flank_low = 0.0;
    double flank_high = 0.0;
    if (!driven && flank == Flank::kMinus) {
      flank_low = contact_parameter;
      flank_high = chi;
    } else if (!driven && flank == Flank::kPlus) {
      flank_low = previous;
      flank_high = contact_parameter;
    } else if (driven && flank == Flank::kMinus) {
      flank_low = previous - mean_pitch_phi;
      flank_high = contact_parameter;
    } else {
      flank_low = contact_parameter;
      flank_high = next + mean_pitch_phi;
    }
    const double fillet_low = std::min(dedendum_parameter, contact_parameter);
    const double fillet_high = std::max(dedendum_parameter, contact_parameter);

    const CurveFunction flank_curve = [this, tooth, flank, driven](double phi) {
      return driven ? driven_flank(tooth, flank, phi)
                    : drive_flank(tooth, flank, phi);
    };
    const CurveFunction fillet_curve = [this, tooth, flank, driven](double phi) {
      return driven ? driven_fillet(tooth, flank, phi)
                    : drive_fillet(tooth, flank, phi);
    };
    const std::vector<CurveIntersection> intersections =
        find_curve_intersections(
            flank_curve,
            std::min(flank_low, flank_high),
            std::max(flank_low, flank_high),
            fillet_curve,
            fillet_low,
            fillet_high);
    const double singular_parameter =
        driven
            ? (flank == Flank::kMinus
                   ? driven_singular_minus[static_cast<std::size_t>(tooth)]
                   : driven_singular_plus[static_cast<std::size_t>(tooth)])
            : (flank == Flank::kMinus
                   ? drive_singular_minus[static_cast<std::size_t>(tooth)]
                   : drive_singular_plus[static_cast<std::size_t>(tooth)]);
    const CurveIntersection selected = choose_intersection(
        intersections,
        [contact_parameter](const CurveIntersection& intersection) {
          return std::abs(intersection.lhs - contact_parameter) > 1e-3 ||
                 std::abs(intersection.rhs - contact_parameter) > 1e-3;
        },
        [singular_parameter](const CurveIntersection& intersection) {
          return std::abs(intersection.lhs - singular_parameter);
        },
        driven ? "driven flank/fillet transition" : "drive flank/fillet transition");
    maximum_intersection_residual =
        std::max(maximum_intersection_residual, selected.residual);
    return selected;
  }

  CurveIntersection solve_addendum_intersection(
      int tooth,
      Flank flank,
      bool driven,
      double transition_parameter) {
    const double chi = chis[static_cast<std::size_t>(tooth)];
    const double previous =
        driven ? previous_driven_chi(tooth) : previous_drive_chi(tooth);
    const double next = driven ? next_driven_chi(tooth) : next_drive_chi(tooth);
    double flank_low = 0.0;
    double flank_high = 0.0;
    double addendum_low = 0.0;
    double addendum_high = 0.0;
    if (!driven && flank == Flank::kMinus) {
      flank_low = transition_parameter;
      flank_high = next;
      addendum_low = previous;
      addendum_high = chi;
    } else if (!driven && flank == Flank::kPlus) {
      flank_low = previous;
      flank_high = transition_parameter;
      addendum_low = chi;
      addendum_high = next;
    } else if (driven && flank == Flank::kMinus) {
      flank_low = previous - mean_pitch_phi;
      flank_high = transition_parameter;
      addendum_low = previous;
      addendum_high = chi;
    } else {
      flank_low = transition_parameter;
      flank_high = next + mean_pitch_phi;
      addendum_low = chi;
      addendum_high = next;
    }
    const CurveFunction flank_curve = [this, tooth, flank, driven](double phi) {
      return driven ? driven_flank(tooth, flank, phi)
                    : drive_flank(tooth, flank, phi);
    };
    const CurveFunction addendum_curve = [this, driven](double phi) {
      return driven ? driven_addendum(phi) : drive_addendum(phi);
    };
    const std::vector<CurveIntersection> intersections = find_curve_intersections(
        flank_curve,
        std::min(flank_low, flank_high),
        std::max(flank_low, flank_high),
        addendum_curve,
        std::min(addendum_low, addendum_high),
        std::max(addendum_low, addendum_high));
    const CurveIntersection selected = choose_intersection(
        intersections,
        [flank_low, flank_high, addendum_low, addendum_high](
            const CurveIntersection& intersection) {
          return intersection.lhs >= std::min(flank_low, flank_high) - 1e-8 &&
                 intersection.lhs <= std::max(flank_low, flank_high) + 1e-8 &&
                 intersection.rhs >= std::min(addendum_low, addendum_high) - 1e-8 &&
                 intersection.rhs <= std::max(addendum_low, addendum_high) + 1e-8;
        },
        [chi](const CurveIntersection& intersection) {
          return std::abs(intersection.lhs - chi) +
                 0.25 * std::abs(intersection.rhs - chi);
        },
        driven ? "driven flank/addendum" : "drive flank/addendum");
    maximum_intersection_residual =
        std::max(maximum_intersection_residual, selected.residual);
    return selected;
  }

  int drive_first_tooth() const {
    return is_closed() ? 0 : padding_teeth;
  }

  int drive_last_tooth() const {
    return drive_first_tooth() + drive_teeth - 1;
  }

  int driven_first_tooth() const {
    return is_closed() ? 0 : padding_teeth;
  }

  int driven_last_tooth() const {
    return driven_first_tooth() + driven_teeth - 1;
  }

  int geometry_first_tooth() const {
    return is_closed() ? 0 : padding_teeth - 1;
  }

  int geometry_last_tooth() const {
    return is_closed() ? static_cast<int>(chis.size()) - 1
                       : padding_teeth + drive_teeth;
  }

  double previous_drive_chi(int tooth) const {
    if (is_closed() && tooth == 0) {
      return chis[static_cast<std::size_t>(drive_teeth - 1)] - drive_cycle;
    }
    return chis[static_cast<std::size_t>(tooth - 1)];
  }

  double next_drive_chi(int tooth) const {
    if (is_closed() && tooth + 1 == drive_teeth) {
      return chis.front() + drive_cycle;
    }
    return chis[static_cast<std::size_t>(tooth + 1)];
  }

  double previous_driven_chi(int tooth) const {
    if (is_closed() && tooth == 0) {
      return chis[static_cast<std::size_t>(driven_teeth - 1)] - driven_cycle;
    }
    return chis[static_cast<std::size_t>(tooth - 1)];
  }

  double next_driven_chi(int tooth) const {
    if (is_closed() && tooth + 1 == driven_teeth) {
      return chis.front() + driven_cycle;
    }
    return chis[static_cast<std::size_t>(tooth + 1)];
  }

  void solve_checkpoints() {
    checkpoints.clear();
    checkpoints.reserve(static_cast<std::size_t>(drive_teeth));
    const std::size_t scalar_count = chis.size();
    drive_singular_minus.resize(scalar_count);
    drive_singular_plus.resize(scalar_count);
    driven_singular_minus.resize(scalar_count);
    driven_singular_plus.resize(scalar_count);
    drive_free_minus.resize(scalar_count);
    drive_free_plus.resize(scalar_count);
    driven_free_minus.resize(scalar_count);
    driven_free_plus.resize(scalar_count);

    for (int tooth = geometry_first_tooth(); tooth <= geometry_last_tooth(); ++tooth) {
      const double drive_minus = singular(tooth, Flank::kMinus, false);
      const double drive_plus = singular(tooth, Flank::kPlus, false);
      const double drive_minus_kappa = drive_kappa(drive_minus);
      const double drive_plus_kappa = drive_kappa(drive_plus);
      const bool drive_minus_free = -drive_minus_kappa <= curvature_limit + 1e-11;
      const bool drive_plus_free = -drive_plus_kappa <= curvature_limit + 1e-11;

      const std::size_t index = static_cast<std::size_t>(tooth);
      drive_singular_minus[index] = drive_minus;
      drive_singular_plus[index] = drive_plus;
      drive_free_minus[index] = drive_minus_free;
      drive_free_plus[index] = drive_plus_free;
      if (tooth >= drive_first_tooth() && tooth <= drive_last_tooth()) {
        checkpoints.push_back({
            chis[index],
            drive_minus,
            drive_plus,
            drive_minus_kappa,
            drive_plus_kappa,
            !drive_minus_free,
            !drive_plus_free,
        });
      }
    }

    const int driven_geometry_first = is_closed() ? 0 : geometry_first_tooth();
    const int driven_geometry_last =
        is_closed() ? driven_teeth - 1 : geometry_last_tooth();
    for (int tooth = driven_geometry_first; tooth <= driven_geometry_last; ++tooth) {
      const double driven_minus = singular(tooth, Flank::kMinus, true);
      const double driven_plus = singular(tooth, Flank::kPlus, true);
      const double driven_minus_kappa = driven_kappa(driven_minus);
      const double driven_plus_kappa = driven_kappa(driven_plus);
      const std::size_t index = static_cast<std::size_t>(tooth);
      driven_singular_minus[index] = driven_minus;
      driven_singular_plus[index] = driven_plus;
      driven_free_minus[index] = driven_minus_kappa <= curvature_limit + 1e-11;
      driven_free_plus[index] = driven_plus_kappa <= curvature_limit + 1e-11;
    }
  }

  void solve_drive_geometry() {
    const double fillet_dedendum_lambda =
        (dedendum - fillet_radius) * std::tan(alpha) +
        fillet_radius / std::cos(alpha);
    const double tangent_lambda =
        ((dedendum - fillet_radius) / std::sin(alpha) + fillet_radius) /
        std::cos(alpha);
    drive_geometry.resize(chis.size());

    for (int tooth = geometry_first_tooth(); tooth <= geometry_last_tooth(); ++tooth) {
      DriveGeometry geometry;
      geometry.minus_fillet_dedendum =
          solve_lambda_target(tooth, Flank::kMinus, fillet_dedendum_lambda, -1);
      geometry.plus_fillet_dedendum =
          solve_lambda_target(tooth, Flank::kPlus, -fillet_dedendum_lambda, 1);

      const std::size_t index = static_cast<std::size_t>(tooth);
      const double minus_contact =
          solve_lambda_target(tooth, Flank::kMinus, tangent_lambda, -1);
      const double plus_contact =
          solve_lambda_target(tooth, Flank::kPlus, -tangent_lambda, 1);
      if (drive_free_minus[index]) {
        geometry.minus_flank_transition = minus_contact;
        geometry.minus_fillet_transition = minus_contact;
      } else {
        const CurveIntersection transition = solve_transition(
            tooth,
            Flank::kMinus,
            false,
            minus_contact,
            geometry.minus_fillet_dedendum);
        geometry.minus_flank_transition = transition.lhs;
        geometry.minus_fillet_transition = transition.rhs;
      }

      if (drive_free_plus[index]) {
        geometry.plus_flank_transition = plus_contact;
        geometry.plus_fillet_transition = plus_contact;
      } else {
        const CurveIntersection transition = solve_transition(
            tooth,
            Flank::kPlus,
            false,
            plus_contact,
            geometry.plus_fillet_dedendum);
        geometry.plus_flank_transition = transition.lhs;
        geometry.plus_fillet_transition = transition.rhs;
      }

      const CurveIntersection minus_addendum = solve_addendum_intersection(
          tooth, Flank::kMinus, false, geometry.minus_flank_transition);
      geometry.minus_flank_addendum = minus_addendum.lhs;
      geometry.minus_addendum = minus_addendum.rhs;
      const CurveIntersection plus_addendum = solve_addendum_intersection(
          tooth, Flank::kPlus, false, geometry.plus_flank_transition);
      geometry.plus_flank_addendum = plus_addendum.lhs;
      geometry.plus_addendum = plus_addendum.rhs;
      drive_geometry[index] = geometry;
    }
  }

  void solve_driven_geometry() {
    const double fillet_dedendum_lambda =
        (dedendum - fillet_radius) * std::tan(alpha) +
        fillet_radius / std::cos(alpha);
    const double tangent_lambda =
        ((dedendum - fillet_radius) / std::sin(alpha) + fillet_radius) /
        std::cos(alpha);
    driven_geometry.resize(chis.size());

    const int first = is_closed() ? 0 : geometry_first_tooth();
    const int last = is_closed() ? driven_teeth - 1 : geometry_last_tooth();
    for (int tooth = first; tooth <= last; ++tooth) {
      DrivenGeometry geometry;
      geometry.minus_fillet_dedendum =
          solve_lambda_target(tooth, Flank::kMinus, -fillet_dedendum_lambda, -1);
      geometry.plus_fillet_dedendum =
          solve_lambda_target(tooth, Flank::kPlus, fillet_dedendum_lambda, 1);

      const std::size_t index = static_cast<std::size_t>(tooth);
      const double minus_contact =
          solve_lambda_target(tooth, Flank::kMinus, -tangent_lambda, 1);
      const double plus_contact =
          solve_lambda_target(tooth, Flank::kPlus, tangent_lambda, -1);
      if (driven_free_minus[index]) {
        geometry.minus_flank_transition = minus_contact;
        geometry.minus_fillet_transition = minus_contact;
      } else {
        const CurveIntersection transition = solve_transition(
            tooth,
            Flank::kMinus,
            true,
            minus_contact,
            geometry.minus_fillet_dedendum);
        geometry.minus_flank_transition = transition.lhs;
        geometry.minus_fillet_transition = transition.rhs;
      }

      if (driven_free_plus[index]) {
        geometry.plus_flank_transition = plus_contact;
        geometry.plus_fillet_transition = plus_contact;
      } else {
        const CurveIntersection transition = solve_transition(
            tooth,
            Flank::kPlus,
            true,
            plus_contact,
            geometry.plus_fillet_dedendum);
        geometry.plus_flank_transition = transition.lhs;
        geometry.plus_fillet_transition = transition.rhs;
      }

      const CurveIntersection minus_addendum = solve_addendum_intersection(
          tooth, Flank::kMinus, true, geometry.minus_flank_transition);
      geometry.minus_flank_addendum = minus_addendum.lhs;
      geometry.minus_addendum = minus_addendum.rhs;
      const CurveIntersection plus_addendum = solve_addendum_intersection(
          tooth, Flank::kPlus, true, geometry.plus_flank_transition);
      geometry.plus_flank_addendum = plus_addendum.lhs;
      geometry.plus_addendum = plus_addendum.rhs;
      driven_geometry[index] = geometry;
    }
  }

  void append_curve(
      std::vector<Point>& outline,
      const CurveFunction& curve,
      double start,
      double end,
      int samples_per_radian) {
    const int samples = std::max(
        4,
        static_cast<int>(
            std::ceil(std::abs(end - start) * static_cast<double>(samples_per_radian))));
    for (int index = 0; index <= samples; ++index) {
      const double fraction = static_cast<double>(index) / static_cast<double>(samples);
      const Point point = to_point(curve(std::lerp(start, end, fraction)));
      if (!outline.empty() && index == 0) {
        maximum_join_gap = std::max(maximum_join_gap, point_distance(outline.back(), point));
      }
      if (outline.empty() || point_distance(outline.back(), point) > 1e-10) {
        outline.push_back(point);
      }
    }
  }

  bool append_clipped_curve(
      std::vector<Point>& outline,
      const CurveFunction& curve,
      double start,
      double end,
      int samples_per_radian) {
    const double low = std::max(std::min(start, end), active_start);
    const double high = std::min(std::max(start, end), active_end);
    if (low >= high) {
      return false;
    }
    std::vector<Point> piece;
    if (start <= end) {
      append_curve(piece, curve, low, high, samples_per_radian);
    } else {
      append_curve(piece, curve, high, low, samples_per_radian);
    }
    if (piece.empty()) {
      return false;
    }
    for (const Point& point : piece) {
      if (outline.empty() || point_distance(outline.back(), point) > 1e-10) {
        outline.push_back(point);
      }
    }
    return true;
  }

  std::vector<Point> build_drive_outline(int samples_per_radian) {
    std::vector<Point> outline;
    const int first = drive_first_tooth();
    const int last = drive_last_tooth();
    for (int tooth = first; tooth <= last; ++tooth) {
      const DriveGeometry& geometry = drive_geometry[static_cast<std::size_t>(tooth)];
      const auto append = [this, &outline, samples_per_radian](
                              const CurveFunction& curve,
                              double start,
                              double end) {
        if (is_closed()) {
          append_curve(outline, curve, start, end, samples_per_radian);
        } else {
          append_clipped_curve(outline, curve, start, end, samples_per_radian);
        }
      };
      append(
          [this, tooth](double phi) {
            return drive_fillet(tooth, Flank::kMinus, phi);
          },
          geometry.minus_fillet_dedendum,
          geometry.minus_fillet_transition);
      append(
          [this, tooth](double phi) {
            return drive_flank(tooth, Flank::kMinus, phi);
          },
          geometry.minus_flank_transition,
          geometry.minus_flank_addendum);
      append(
          [this](double phi) { return drive_addendum(phi); },
          geometry.minus_addendum,
          geometry.plus_addendum);
      append(
          [this, tooth](double phi) {
            return drive_flank(tooth, Flank::kPlus, phi);
          },
          geometry.plus_flank_addendum,
          geometry.plus_flank_transition);
      append(
          [this, tooth](double phi) {
            return drive_fillet(tooth, Flank::kPlus, phi);
          },
          geometry.plus_fillet_transition,
          geometry.plus_fillet_dedendum);

      const int next_tooth =
          is_closed() ? (tooth + 1) % drive_teeth : tooth + 1;
      double next_parameter =
          drive_geometry[static_cast<std::size_t>(next_tooth)].minus_fillet_dedendum;
      if (is_closed() && next_tooth == 0) {
        next_parameter += drive_cycle;
      }
      append(
          [this](double phi) { return drive_dedendum(phi); },
          geometry.plus_fillet_dedendum,
          next_parameter);
    }
    if (is_closed()) {
      close_outline(outline);
    } else {
      close_open_outline(
          outline,
          [this](double phi) { return 0.25 * drive_centrode(phi); },
          samples_per_radian);
    }
    return outline;
  }

  std::vector<Point> build_driven_outline(int samples_per_radian) {
    std::vector<Point> outline;
    const int first = driven_first_tooth();
    const int last = driven_last_tooth();
    for (int tooth = first; tooth <= last; ++tooth) {
      const DrivenGeometry& geometry = driven_geometry[static_cast<std::size_t>(tooth)];
      const auto append = [this, &outline, samples_per_radian](
                              const CurveFunction& curve,
                              double start,
                              double end) {
        if (is_closed()) {
          append_curve(outline, curve, start, end, samples_per_radian);
        } else {
          append_clipped_curve(outline, curve, start, end, samples_per_radian);
        }
      };
      append(
          [this, tooth](double phi) {
            return driven_flank(tooth, Flank::kMinus, phi);
          },
          geometry.minus_flank_addendum,
          geometry.minus_flank_transition);
      append(
          [this, tooth](double phi) {
            return driven_fillet(tooth, Flank::kMinus, phi);
          },
          geometry.minus_fillet_transition,
          geometry.minus_fillet_dedendum);
      append(
          [this](double phi) { return driven_dedendum(phi); },
          geometry.minus_fillet_dedendum,
          geometry.plus_fillet_dedendum);
      append(
          [this, tooth](double phi) {
            return driven_fillet(tooth, Flank::kPlus, phi);
          },
          geometry.plus_fillet_dedendum,
          geometry.plus_fillet_transition);
      append(
          [this, tooth](double phi) {
            return driven_flank(tooth, Flank::kPlus, phi);
          },
          geometry.plus_flank_transition,
          geometry.plus_flank_addendum);

      const int next_tooth =
          is_closed() ? (tooth + 1) % driven_teeth : tooth + 1;
      double next_parameter =
          driven_geometry[static_cast<std::size_t>(next_tooth)].minus_addendum;
      if (is_closed() && next_tooth == 0) {
        next_parameter += driven_cycle;
      }
      append(
          [this](double phi) { return driven_addendum(phi); },
          geometry.plus_addendum,
          next_parameter);
    }
    if (is_closed()) {
      close_outline(outline);
    } else {
      close_open_outline(
          outline,
          [this](double phi) { return 0.25 * driven_centrode(phi); },
          samples_per_radian);
    }
    return outline;
  }

  void close_outline(std::vector<Point>& outline) {
    if (outline.size() < 3) {
      throw std::runtime_error("Generated outline has too few points.");
    }
    const double seam_gap = point_distance(outline.back(), outline.front());
    maximum_join_gap = std::max(maximum_join_gap, seam_gap);
    if (seam_gap <= 2e-6) {
      outline.back() = outline.front();
    } else {
      outline.push_back(outline.front());
    }
  }

  void close_open_outline(
      std::vector<Point>& outline,
      const CurveFunction& inner_boundary,
      int samples_per_radian) {
    if (outline.size() < 3) {
      throw std::runtime_error("Generated open outline has too few points.");
    }
    std::vector<Point> backing;
    append_curve(
        backing,
        inner_boundary,
        active_end,
        active_start,
        samples_per_radian);
    for (const Point& point : backing) {
      if (point_distance(outline.back(), point) > 1e-10) {
        outline.push_back(point);
      }
    }
    if (point_distance(outline.back(), outline.front()) > 1e-12) {
      outline.push_back(outline.front());
    } else {
      outline.back() = outline.front();
    }
  }

  GenerationResult run(int samples_per_radian) {
    if (samples_per_radian < 20) {
      throw std::invalid_argument("samples_per_radian must be at least 20.");
    }
    validate_centrodes();
    solve_chis();
    solve_checkpoints();
    solve_drive_geometry();
    solve_driven_geometry();
    std::vector<Point> drive = build_drive_outline(samples_per_radian);
    std::vector<Point> driven = build_driven_outline(samples_per_radian);

    if (!is_simple_closed_polygon(drive)) {
      throw std::runtime_error(
          "Drive outline is not a simple CGAL polygon: " +
          self_intersection_description(drive));
    }
    if (!is_simple_closed_polygon(driven)) {
      throw std::runtime_error(
          "Driven outline is not a simple CGAL polygon: " +
          self_intersection_description(driven));
    }
    if (std::abs(signed_polygon_area(drive)) < 1e-8 ||
        std::abs(signed_polygon_area(driven)) < 1e-8) {
      throw std::runtime_error("Generated outline has zero area.");
    }
    if (maximum_join_gap > 2e-6) {
      throw std::runtime_error("Generated curve pieces do not join continuously.");
    }
    double overlap_area = 0.0;
    const int phase_count = is_closed() ? 12 : 8;
    for (int phase_index = 0; phase_index < phase_count; ++phase_index) {
      const double fraction =
          static_cast<double>(phase_index) / static_cast<double>(phase_count);
      const double phi = std::lerp(active_start, active_end, fraction);
      overlap_area = std::max(
          overlap_area,
          placed_pair_overlap_area(
              drive,
              driven,
              center_distance,
              phi - active_start,
              -(psi(phi) - psi(active_start))));
    }
    const double overlap_tolerance =
        is_closed()
            ? 1e-10
            : 0.25 * config.module * config.module /
                  std::pow(static_cast<double>(samples_per_radian), 2);
    if (overlap_area > overlap_tolerance) {
      std::ostringstream message;
      message << "Placed gear solids overlap by area " << overlap_area
              << ", exceeding the discretization tolerance "
              << overlap_tolerance << ".";
      throw std::runtime_error(message.str());
    }

    return {
        config,
        drive_teeth,
        driven_teeth,
        average_angular_ratio,
        total_integral,
        center_distance,
        curvature_limit,
        maximum_join_gap,
        maximum_intersection_residual,
        overlap_area,
        checkpoints,
        std::move(drive),
        std::move(driven),
    };
  }

  SampleConfig config;
  double alpha = 0.0;
  double addendum = 0.0;
  double dedendum = 0.0;
  double fillet_radius = 0.0;
  IntegralTable integral;
  int drive_teeth = 0;
  int driven_teeth = 0;
  int padding_teeth = 6;
  double active_start = 0.0;
  double active_end = 0.0;
  double drive_cycle = 0.0;
  double driven_cycle = 0.0;
  double average_angular_ratio = 1.0;
  double total_integral = 0.0;
  double center_distance = 0.0;
  double mean_pitch_phi = 0.0;
  double curvature_limit = 0.0;
  double maximum_join_gap = 0.0;
  double maximum_intersection_residual = 0.0;
  std::vector<double> chis;
  std::vector<double> drive_singular_minus;
  std::vector<double> drive_singular_plus;
  std::vector<double> driven_singular_minus;
  std::vector<double> driven_singular_plus;
  std::vector<bool> drive_free_minus;
  std::vector<bool> drive_free_plus;
  std::vector<bool> driven_free_minus;
  std::vector<bool> driven_free_plus;
  std::vector<ToothCheckpoint> checkpoints;
  std::vector<DriveGeometry> drive_geometry;
  std::vector<DrivenGeometry> driven_geometry;
};

std::vector<SampleConfig> builtin_samples() {
  return {
      {
          "paper",
          "Section 8: psi(phi) = phi - (2 - sqrt(2)) sin(phi)",
          {{1, -(2.0 - std::sqrt(2.0))}},
          14,
          2.0,
          20.0,
          1.0,
          1.2,
          0.3,
          GearTopology::kClosed,
          {},
      },
      {
          "two_lobe",
          "psi(phi) = phi - 0.12 sin(2 phi)",
          {{2, -0.12}},
          24,
          1.0,
          20.0,
          1.0,
          1.2,
          0.3,
          GearTopology::kClosed,
          {},
      },
      {
          "asymmetric",
          "psi(phi) = phi - 0.12 sin(phi) - 0.025 sin(2 phi)",
          {{1, -0.12}, {2, -0.025}},
          24,
          1.0,
          20.0,
          1.0,
          1.2,
          0.3,
          GearTopology::kClosed,
          {},
      },
      {
          "three_lobe",
          "psi(phi) = phi - 0.055 sin(3 phi)",
          {{3, -0.055}},
          30,
          1.0,
          20.0,
          1.0,
          1.2,
          0.3,
          GearTopology::kClosed,
          {},
      },
  };
}

SampleConfig builtin_sample(const std::string& name) {
  const std::vector<SampleConfig> samples = builtin_samples();
  const auto iterator = std::find_if(
      samples.begin(),
      samples.end(),
      [&name](const SampleConfig& sample) { return sample.name == name; });
  if (iterator == samples.end()) {
    throw std::invalid_argument("Unknown sample: " + name);
  }
  return *iterator;
}

GearGenerator::GearGenerator(SampleConfig config) : config_(std::move(config)) {}

GenerationResult GearGenerator::generate(int samples_per_radian) {
  Impl implementation(config_);
  return implementation.run(samples_per_radian);
}

bool is_simple_closed_polygon(const std::vector<Point>& points) {
  if (points.size() < 4 || point_distance(points.front(), points.back()) > 1e-8) {
    return false;
  }
  CGAL::Polygon_2<Kernel> polygon(points.begin(), points.end() - 1);
  return polygon.is_simple();
}

double signed_polygon_area(const std::vector<Point>& points) {
  if (points.size() < 4) {
    return 0.0;
  }
  CGAL::Polygon_2<Kernel> polygon(points.begin(), points.end() - 1);
  return CGAL::to_double(polygon.area());
}

void write_result(const GenerationResult& result, const std::filesystem::path& directory) {
  std::filesystem::create_directories(directory);

  const auto write_csv = [](const std::filesystem::path& path,
                            const std::vector<Point>& outline) {
    std::ofstream output(path);
    if (!output) {
      throw std::runtime_error("Unable to write " + path.string());
    }
    output << "x,y\n" << std::setprecision(17);
    for (const Point& point : outline) {
      output << point.x() << ',' << point.y() << '\n';
    }
  };
  write_csv(directory / "drive.csv", result.drive_outline);
  write_csv(directory / "driven.csv", result.driven_outline);

  std::ofstream metadata(directory / "metadata.json");
  if (!metadata) {
    throw std::runtime_error("Unable to write metadata.");
  }
  metadata << std::setprecision(17)
           << "{\n"
           << "  \"name\": \"" << json_escape(result.config.name) << "\",\n"
           << "  \"description\": \"" << json_escape(result.config.description) << "\",\n"
           << "  \"topology\": \""
           << (result.config.topology == GearTopology::kClosed ? "closed" : "open")
           << "\",\n"
           << "  \"drive_teeth\": " << result.drive_teeth << ",\n"
           << "  \"driven_teeth\": " << result.driven_teeth << ",\n"
           << "  \"average_angular_ratio\": " << result.average_angular_ratio << ",\n"
           << "  \"module\": " << result.config.module << ",\n"
           << "  \"pressure_angle_deg\": " << result.config.pressure_angle_deg << ",\n"
           << "  \"total_integral\": " << result.total_integral << ",\n"
           << "  \"center_distance\": " << result.center_distance << ",\n"
           << "  \"undercut_curvature_limit\": " << result.undercut_curvature_limit << ",\n"
           << "  \"maximum_join_gap\": " << result.maximum_join_gap << ",\n"
           << "  \"maximum_intersection_residual\": "
           << result.maximum_intersection_residual << ",\n"
           << "  \"placed_pair_overlap_area\": " << result.placed_pair_overlap_area << ",\n"
           << "  \"drive_area\": " << signed_polygon_area(result.drive_outline) << ",\n"
           << "  \"driven_area\": " << signed_polygon_area(result.driven_outline) << "\n"
           << "}\n";
}

}  // namespace ncgear
