#include "ncgear/gear.hpp"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_set_2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace ncgear {
namespace {

using Complex = std::complex<double>;
using ExactKernel = CGAL::Exact_predicates_exact_constructions_kernel;
using ExactPoint = ExactKernel::Point_2;
using ExactPolygon = CGAL::Polygon_2<ExactKernel>;
using ExactPolygonSet = CGAL::Polygon_set_2<ExactKernel>;
using ExactPolygonWithHoles = CGAL::Polygon_with_holes_2<ExactKernel>;
using ScalarFunction = std::function<double(double)>;

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

Complex cis(double angle) {
  return {std::cos(angle), std::sin(angle)};
}

ExactPoint to_exact_point(const Complex& value) {
  return {value.real(), value.imag()};
}

double point_distance(const Point& lhs, const Point& rhs) {
  return std::hypot(lhs.x() - rhs.x(), lhs.y() - rhs.y());
}

double integrate_gauss8(
    const ScalarFunction& function,
    double start,
    double end) {
  if (start == end) {
    return 0.0;
  }
  if (end < start) {
    return -integrate_gauss8(function, end, start);
  }
  const double midpoint = 0.5 * (start + end);
  const double radius = 0.5 * (end - start);
  double sum = 0.0;
  for (std::size_t index = 0; index < kGaussNodes.size(); ++index) {
    sum +=
        kGaussWeights[index] * function(midpoint + radius * kGaussNodes[index]);
  }
  return radius * sum;
}

class IntegralTable {
 public:
  IntegralTable(
      ScalarFunction function,
      double domain_start,
      double domain_end,
      bool periodic,
      int interval_count = 1 << 15)
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
          prefix_[static_cast<std::size_t>(index)] +
          integrate_gauss8(function_, start, start + step_);
    }
    domain_integral_ = prefix_.back();
  }

  double integral(double start, double end) const {
    return antiderivative(end) - antiderivative(start);
  }

 private:
  double antiderivative(double value) const {
    double cycles = 0.0;
    double wrapped = value;
    const double length = domain_end_ - domain_start_;
    if (periodic_) {
      cycles = std::floor((value - domain_start_) / length);
      wrapped = value - cycles * length;
      if (wrapped < domain_start_) {
        wrapped += length;
      } else if (wrapped >= domain_end_) {
        wrapped -= length;
      }
    } else {
      if (value < domain_start_ - 1e-10 || value > domain_end_ + 1e-10) {
        throw std::out_of_range("Integral query is outside the motion domain.");
      }
      wrapped = std::clamp(value, domain_start_, domain_end_);
    }

    const int interval = std::clamp(
        static_cast<int>(std::floor((wrapped - domain_start_) / step_)),
        0,
        static_cast<int>(prefix_.size()) - 2);
    const double interval_start =
        domain_start_ + static_cast<double>(interval) * step_;
    return cycles * domain_integral_ +
           prefix_[static_cast<std::size_t>(interval)] +
           integrate_gauss8(function_, interval_start, wrapped);
  }

  ScalarFunction function_;
  double domain_start_ = 0.0;
  double domain_end_ = 0.0;
  bool periodic_ = false;
  double step_ = 0.0;
  double domain_integral_ = 0.0;
  std::vector<double> prefix_;
};

class MotionLaw {
 public:
  explicit MotionLaw(const SampleConfig& config)
      : config_(config),
        sampled_(!config.transmission.psi.empty()),
        closed_(config.topology == GearTopology::kClosed) {}

  double value(double phi, int derivative) const {
    if (!sampled_) {
      return analytic(phi, derivative);
    }
    return quintic(phi, derivative);
  }

 private:
  double analytic(double phi, int derivative) const {
    double result = derivative == 0 ? phi : derivative == 1 ? 1.0 : 0.0;
    for (const Harmonic& harmonic : config_.harmonics) {
      const double order = static_cast<double>(harmonic.order);
      const double phase = order * phi +
                           static_cast<double>(derivative) * 0.5 * kPi;
      result += harmonic.amplitude * std::pow(order, derivative) *
                std::sin(phase);
    }
    return result;
  }

  double quintic(double phi, int derivative) const {
    const TransmissionSamples& samples = config_.transmission;
    const std::size_t count = samples.psi.size();
    const double domain_length = samples.domain_end - samples.domain_start;
    const double interpolation_length = closed_ ? samples.period : domain_length;
    const double step =
        closed_ ? interpolation_length / static_cast<double>(count)
                : interpolation_length / static_cast<double>(count - 1);

    double cycles = 0.0;
    double x = phi;
    if (closed_) {
      cycles = std::floor((phi - samples.domain_start) / samples.period);
      x = phi - cycles * samples.period;
      if (x < samples.domain_start) {
        x += samples.period;
      } else if (x >= samples.domain_start + samples.period) {
        x -= samples.period;
      }
    } else {
      x = std::clamp(x, samples.domain_start, samples.domain_end);
    }

    const double scaled = (x - samples.domain_start) / step;
    int index = static_cast<int>(std::floor(scaled));
    double t = scaled - static_cast<double>(index);
    if (!closed_ && index >= static_cast<int>(count) - 1) {
      index = static_cast<int>(count) - 2;
      t = 1.0;
    }
    index = std::clamp(index, 0, static_cast<int>(count) - 1);
    const int next =
        closed_ ? (index + 1) % static_cast<int>(count) : index + 1;

    const double y0 = samples.psi[static_cast<std::size_t>(index)];
    double y1 = samples.psi[static_cast<std::size_t>(next)];
    if (closed_ && next == 0) {
      y1 += samples.cycle_delta;
    }
    const double d0 = samples.psi1[static_cast<std::size_t>(index)];
    const double d1 = samples.psi1[static_cast<std::size_t>(next)];
    const double dd0 = samples.psi2[static_cast<std::size_t>(index)];
    const double dd1 = samples.psi2[static_cast<std::size_t>(next)];

    const double a0 = y0;
    const double a1 = step * d0;
    const double a2 = 0.5 * step * step * dd0;
    const double endpoint_value_error = y1 - (a0 + a1 + a2);
    const double endpoint_slope_error = step * d1 - (a1 + 2.0 * a2);
    const double endpoint_curvature_error =
        step * step * dd1 - 2.0 * a2;
    const double a3 =
        10.0 * endpoint_value_error - 4.0 * endpoint_slope_error +
        0.5 * endpoint_curvature_error;
    const double a4 =
        -15.0 * endpoint_value_error + 7.0 * endpoint_slope_error -
        endpoint_curvature_error;
    const double a5 =
        6.0 * endpoint_value_error - 3.0 * endpoint_slope_error +
        0.5 * endpoint_curvature_error;

    double result = 0.0;
    if (derivative == 0) {
      result =
          a0 + t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
      result += cycles * samples.cycle_delta;
    } else if (derivative == 1) {
      result =
          (a1 + t * (2.0 * a2 +
                     t * (3.0 * a3 + t * (4.0 * a4 + t * 5.0 * a5)))) /
          step;
    } else if (derivative == 2) {
      result =
          (2.0 * a2 +
           t * (6.0 * a3 + t * (12.0 * a4 + t * 20.0 * a5))) /
          (step * step);
    } else if (derivative == 3) {
      result =
          (6.0 * a3 + t * (24.0 * a4 + t * 60.0 * a5)) /
          (step * step * step);
    } else {
      throw std::invalid_argument("Motion derivative order must be between 0 and 3.");
    }
    return result;
  }

  const SampleConfig& config_;
  bool sampled_ = false;
  bool closed_ = true;
};

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

double exact_component_area(const ExactPolygonWithHoles& component) {
  ExactKernel::FT area = CGAL::abs(component.outer_boundary().area());
  for (auto hole = component.holes_begin(); hole != component.holes_end(); ++hole) {
    area -= CGAL::abs(hole->area());
  }
  return CGAL::to_double(area);
}

ExactPolygon make_circle_polygon(double radius, int point_count) {
  ExactPolygon polygon;
  for (int index = 0; index < point_count; ++index) {
    const double angle =
        2.0 * kPi * static_cast<double>(index) /
        static_cast<double>(point_count);
    polygon.push_back(ExactPoint(radius * std::cos(angle), radius * std::sin(angle)));
  }
  return polygon;
}

ExactPolygon make_open_sector(
    double start_angle,
    double end_angle,
    double inner_radius,
    double outer_radius) {
  const double span = end_angle - start_angle;
  const int arc_count =
      std::max(16, static_cast<int>(std::ceil(std::abs(span) * 160.0)));
  ExactPolygon polygon;
  for (int index = 0; index <= arc_count; ++index) {
    const double fraction =
        static_cast<double>(index) / static_cast<double>(arc_count);
    const double angle = std::lerp(start_angle, end_angle, fraction);
    polygon.push_back(
        ExactPoint(outer_radius * std::cos(angle), outer_radius * std::sin(angle)));
  }
  for (int index = arc_count; index >= 0; --index) {
    const double fraction =
        static_cast<double>(index) / static_cast<double>(arc_count);
    const double angle = std::lerp(start_angle, end_angle, fraction);
    polygon.push_back(
        ExactPoint(inner_radius * std::cos(angle), inner_radius * std::sin(angle)));
  }
  if (polygon.orientation() == CGAL::CLOCKWISE) {
    polygon.reverse_orientation();
  }
  return polygon;
}

std::vector<Point> simplify_outline(
    const ExactPolygon& polygon,
    double tolerance) {
  std::vector<Point> points;
  points.reserve(polygon.size() + 1);
  for (const ExactPoint& point : polygon) {
    const Point converted(CGAL::to_double(point.x()), CGAL::to_double(point.y()));
    if (points.empty() || point_distance(points.back(), converted) > tolerance * 0.05) {
      points.push_back(converted);
    }
  }

  bool changed = true;
  while (changed && points.size() > 3) {
    changed = false;
    std::vector<Point> filtered;
    filtered.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
      const Point& previous =
          points[(index + points.size() - 1) % points.size()];
      const Point& current = points[index];
      const Point& next = points[(index + 1) % points.size()];
      const double ax = current.x() - previous.x();
      const double ay = current.y() - previous.y();
      const double bx = next.x() - current.x();
      const double by = next.y() - current.y();
      const double lengths = std::hypot(ax, ay) + std::hypot(bx, by);
      const double twice_area = std::abs(ax * by - ay * bx);
      const bool forward = ax * bx + ay * by >= 0.0;
      if (forward && lengths > 0.0 && twice_area / lengths < tolerance * 0.02) {
        changed = true;
      } else {
        filtered.push_back(current);
      }
    }
    points = std::move(filtered);
  }

  if (points.size() < 3) {
    throw std::runtime_error("Swept cutter left fewer than three boundary points.");
  }
  points.push_back(points.front());
  return points;
}

double point_segment_distance(
    const Point& point,
    const Point& start,
    const Point& end) {
  const double dx = end.x() - start.x();
  const double dy = end.y() - start.y();
  const double denominator = dx * dx + dy * dy;
  if (denominator <= 0.0) {
    return point_distance(point, start);
  }
  const double fraction = std::clamp(
      ((point.x() - start.x()) * dx + (point.y() - start.y()) * dy) /
          denominator,
      0.0,
      1.0);
  return std::hypot(
      point.x() - std::lerp(start.x(), end.x(), fraction),
      point.y() - std::lerp(start.y(), end.y(), fraction));
}

void simplify_chain_recursive(
    const std::vector<Point>& points,
    std::size_t first,
    std::size_t last,
    double tolerance,
    std::vector<bool>& keep) {
  if (last <= first + 1) {
    return;
  }
  double maximum_distance = 0.0;
  std::size_t maximum_index = first;
  for (std::size_t index = first + 1; index < last; ++index) {
    const double distance =
        point_segment_distance(points[index], points[first], points[last]);
    if (distance > maximum_distance) {
      maximum_distance = distance;
      maximum_index = index;
    }
  }
  if (maximum_distance <= tolerance) {
    return;
  }
  keep[maximum_index] = true;
  simplify_chain_recursive(points, first, maximum_index, tolerance, keep);
  simplify_chain_recursive(points, maximum_index, last, tolerance, keep);
}

std::vector<Point> simplify_closed_for_sweep(
    const std::vector<Point>& closed,
    double tolerance) {
  const std::size_t count = closed.size() - 1;
  std::size_t split = 1;
  double split_distance = 0.0;
  for (std::size_t index = 1; index < count; ++index) {
    const double distance = point_distance(closed.front(), closed[index]);
    if (distance > split_distance) {
      split_distance = distance;
      split = index;
    }
  }
  std::vector<bool> keep(count, false);
  keep[0] = true;
  keep[split] = true;
  simplify_chain_recursive(closed, 0, split, tolerance, keep);

  std::vector<Point> wrapped;
  wrapped.reserve(count - split + 1);
  for (std::size_t index = split; index < count; ++index) {
    wrapped.push_back(closed[index]);
  }
  wrapped.push_back(closed.front());
  std::vector<bool> wrapped_keep(wrapped.size(), false);
  wrapped_keep.front() = true;
  wrapped_keep.back() = true;
  simplify_chain_recursive(
      wrapped, 0, wrapped.size() - 1, tolerance, wrapped_keep);
  for (std::size_t index = 1; index + 1 < wrapped.size(); ++index) {
    if (wrapped_keep[index]) {
      keep[split + index] = true;
    }
  }

  std::vector<Point> result;
  for (std::size_t index = 0; index < count; ++index) {
    if (keep[index]) {
      result.push_back(closed[index]);
    }
  }
  result.push_back(result.front());
  return result;
}

ExactPolygon transform_rack_polygon(
    const std::vector<Complex>& local,
    const Complex& pitch_point,
    const Complex& tangent,
    const Complex& outward_normal) {
  ExactPolygon polygon;
  std::optional<Complex> previous;
  for (const Complex& point : local) {
    if (previous && std::abs(point - *previous) <= 1e-12) {
      continue;
    }
    polygon.push_back(to_exact_point(
        pitch_point + point.real() * tangent + point.imag() * outward_normal));
    previous = point;
  }
  if (polygon.orientation() == CGAL::CLOCKWISE) {
    polygon.reverse_orientation();
  }
  return polygon;
}

ExactPolygon transform_polygon(
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
}

double polygon_set_area(const ExactPolygonSet& set) {
  std::list<ExactPolygonWithHoles> components;
  set.polygons_with_holes(std::back_inserter(components));
  return std::accumulate(
      components.begin(),
      components.end(),
      0.0,
      [](double value, const ExactPolygonWithHoles& component) {
        return value + exact_component_area(component);
      });
}

double placed_pair_overlap_area(
    const ExactPolygonSet& placed_drive,
    const std::vector<Point>& driven,
    double center_distance,
    double driven_angle) {
  const ExactPolygon driven_polygon =
      transform_polygon(driven, driven_angle, center_distance);
  ExactPolygonSet overlap(placed_drive);
  overlap.intersection(driven_polygon);
  return polygon_set_area(overlap);
}

double placed_pair_overlap_area(
    const std::vector<Point>& drive,
    const std::vector<Point>& driven,
    double center_distance,
    double drive_angle,
    double driven_angle) {
  const ExactPolygon drive_polygon =
      transform_polygon(drive, drive_angle, 0.0);
  ExactPolygonSet placed_drive;
  placed_drive.insert(drive_polygon);
  return placed_pair_overlap_area(
      placed_drive, driven, center_distance, driven_angle);
}

template <typename Function>
std::vector<double> parallel_values(int count, Function function) {
  // Exact overlap checks are independent and expensive. Keep concurrency
  // bounded to avoid multiplying CGAL's peak arrangement memory.
  constexpr int kMaximumWorkers = 8;
  std::vector<double> values(static_cast<std::size_t>(count));
  const unsigned hardware_threads = std::thread::hardware_concurrency();
  const int available_threads = static_cast<int>(
      hardware_threads == 0 ? 1 : hardware_threads);
  const int worker_count = std::min(
      count,
      std::min(kMaximumWorkers, available_threads));
  const int block_size = (count + worker_count - 1) / worker_count;
  const auto evaluate_block = [&](int begin, int end) {
    for (int index = begin; index < end; ++index) {
      values[static_cast<std::size_t>(index)] = function(index);
    }
  };

  std::vector<std::future<void>> workers;
  workers.reserve(static_cast<std::size_t>(worker_count - 1));
  for (int worker = 1; worker < worker_count; ++worker) {
    const int begin = worker * block_size;
    const int end = std::min(count, begin + block_size);
    if (begin >= end) {
      break;
    }
    try {
      workers.push_back(std::async(
          std::launch::async,
          [&, begin, end]() { evaluate_block(begin, end); }));
    } catch (const std::system_error&) {
      evaluate_block(begin, end);
    }
  }
  evaluate_block(0, std::min(count, block_size));
  for (std::future<void>& worker : workers) {
    worker.get();
  }
  return values;
}

}  // namespace

struct GearGenerator::Impl {
  explicit Impl(SampleConfig sample)
      : config(std::move(sample)),
        motion(config),
        alpha(config.pressure_angle_deg * kPi / 180.0),
        addendum(config.addendum_factor * config.module),
        dedendum(config.dedendum_factor * config.module),
        fillet_radius(config.fillet_factor * config.module),
        integral(
            [this](double phi) { return arc_density(phi); },
            domain_start(),
            domain_end(),
            is_closed()) {
    validate_config();
    drive_teeth = config.teeth;
    active_start = is_closed() ? domain_start() : config.transmission.active_start;
    active_end =
        is_closed() ? domain_start() + period() : config.transmission.active_end;
    total_integral = integral.integral(active_start, active_end);
    center_distance =
        static_cast<double>(drive_teeth) * kPi * config.module / total_integral;
    if (is_closed()) {
      const double exact_driven_teeth =
          static_cast<double>(drive_teeth) * period() / cycle_delta();
      driven_teeth = static_cast<int>(std::lround(exact_driven_teeth));
      if (driven_teeth <= 0 ||
          std::abs(exact_driven_teeth - static_cast<double>(driven_teeth)) > 1e-7) {
        throw std::invalid_argument(
            "Closed transmission requires an integral driven tooth count.");
      }
      drive_cycle = period();
      driven_cycle =
          period() * static_cast<double>(driven_teeth) /
          static_cast<double>(drive_teeth);
      validate_closed_motion();
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

  bool is_closed() const {
    return config.topology == GearTopology::kClosed;
  }

  bool has_sampled_motion() const {
    return !config.transmission.psi.empty();
  }

  double domain_start() const {
    return has_sampled_motion() ? config.transmission.domain_start : 0.0;
  }

  double domain_end() const {
    return has_sampled_motion() ? config.transmission.domain_end : 2.0 * kPi;
  }

  double period() const {
    return has_sampled_motion() ? config.transmission.period : 2.0 * kPi;
  }

  double cycle_delta() const {
    return has_sampled_motion() ? config.transmission.cycle_delta : 2.0 * kPi;
  }

  double psi(double phi) const {
    return motion.value(phi, 0);
  }

  double psi1(double phi) const {
    return motion.value(phi, 1);
  }

  double psi2(double phi) const {
    return motion.value(phi, 2);
  }

  double psi3(double phi) const {
    return motion.value(phi, 3);
  }

  void validate_config() const {
    if (config.name.empty() || config.teeth < 6 || config.module <= 0.0) {
      throw std::invalid_argument("Invalid sample name, tooth count, or module.");
    }
    if (!(alpha > 0.0 && alpha < 0.45 * kPi)) {
      throw std::invalid_argument("Pressure angle must be between 0 and 81 degrees.");
    }
    if (!(addendum > 0.0 && dedendum > fillet_radius &&
          fillet_radius >= 0.0)) {
      throw std::invalid_argument("Invalid cutter dimensions.");
    }
    const double pitch = kPi * config.module;
    if (0.25 * pitch - dedendum * std::tan(alpha) <= 0.03 * config.module) {
      throw std::invalid_argument(
          "Rack cutter tip is too narrow; reduce dedendum or pressure angle.");
    }

    if (has_sampled_motion()) {
      const std::size_t count = config.transmission.psi.size();
      if (count < 64 || config.transmission.psi1.size() != count ||
          config.transmission.psi2.size() != count) {
        throw std::invalid_argument(
            "Sampled motion requires equal psi, psi1 and psi2 arrays of at least 64 values.");
      }
      if (!(domain_start() < domain_end())) {
        throw std::invalid_argument("Sampled motion domain is invalid.");
      }
      if (!is_closed() &&
          !(domain_start() < config.transmission.active_start &&
            config.transmission.active_start < config.transmission.active_end &&
            config.transmission.active_end < domain_end())) {
        throw std::invalid_argument(
            "Open motion requires padding on both sides of the active interval.");
      }
    } else if (!is_closed()) {
      throw std::invalid_argument("Open gears require sampled motion data.");
    }

    const int check_count = has_sampled_motion()
                                ? std::max(
                                      8192,
                                      static_cast<int>(
                                          config.transmission.psi.size() * 2))
                                : 16384;
    for (int index = 0; index <= check_count; ++index) {
      const double phi = std::lerp(
          domain_start(),
          domain_end(),
          static_cast<double>(index) / static_cast<double>(check_count));
      const double ratio = psi1(phi);
      if (!std::isfinite(ratio) || ratio <= 1e-7 || ratio >= 1e7) {
        throw std::invalid_argument(
            "Motion law is not a bounded orientation-preserving map.");
      }
    }
  }

  void validate_closed_motion() const {
    const double scale = std::max(1.0, std::abs(cycle_delta()));
    for (int index = 0; index < 4096; ++index) {
      const double phi =
          active_start + period() * static_cast<double>(index) / 4096.0;
      const double shifted = phi + driven_cycle;
      if (std::abs((psi(shifted) - psi(phi)) - 2.0 * kPi) >
              3e-5 * scale ||
          std::abs(psi1(shifted) - psi1(phi)) > 3e-5 ||
          std::abs(psi2(shifted) - psi2(phi)) > 1e-4) {
        throw std::invalid_argument(
            "Closed motion is not compatible with one driven-gear revolution.");
      }
    }
  }

  double w(double phi) const {
    const double first = psi1(phi);
    return std::hypot(psi2(phi), first * (1.0 + first));
  }

  double arc_density(double phi) const {
    const double first = psi1(phi);
    return w(phi) / std::pow(1.0 + first, 2);
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
    return (1.0 + first) *
           (first * (psi3(phi) - first - first * first) -
            2.0 * second * second) /
           std::pow(w(phi), 2);
  }

  double driven_h(double phi) const {
    const double first = psi1(phi);
    const double second = psi2(phi);
    return (1.0 + first) *
           (first * (psi3(phi) + first * first + first * first * first) -
            second * second) /
           std::pow(w(phi), 2);
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

  void measure_centrodes() {
    maximum_drive_curvature = -std::numeric_limits<double>::infinity();
    minimum_driven_curvature = std::numeric_limits<double>::infinity();
    maximum_pitch_radius = 0.0;
    minimum_pitch_radius = std::numeric_limits<double>::infinity();
    const double span = is_closed() ? std::max(drive_cycle, driven_cycle)
                                    : domain_end() - domain_start();
    const double start = is_closed() ? active_start : domain_start();
    for (int index = 0; index <= 16384; ++index) {
      const double phi =
          start + span * static_cast<double>(index) / 16384.0;
      maximum_drive_curvature =
          std::max(maximum_drive_curvature, drive_kappa(phi));
      minimum_driven_curvature =
          std::min(minimum_driven_curvature, driven_kappa(phi));
      maximum_pitch_radius = std::max(
          {maximum_pitch_radius, drive_radius(phi), driven_radius(phi)});
      minimum_pitch_radius = std::min(
          {minimum_pitch_radius, drive_radius(phi), driven_radius(phi)});
    }
    centrodes_are_convex =
        maximum_drive_curvature <= 1e-9 && minimum_driven_curvature >= -1e-9;
  }

  void append_involute_tooth(
      std::vector<Complex>& boundary,
      double center,
      double margin) const {
    const double pitch = kPi * config.module;
    const double tangent = std::tan(alpha);
    const double root_half_width = 0.25 * pitch + addendum * tangent;
    const double sharp_tip_half_width = 0.25 * pitch - dedendum * tangent;
    const double maximum_radius =
        0.9 * sharp_tip_half_width * std::cos(alpha) /
        (1.0 - std::sin(alpha));
    const double radius = std::clamp(fillet_radius, 0.0, maximum_radius);
    const double y_shift = -margin;

    boundary.emplace_back(center - 0.5 * pitch, addendum + y_shift);
    boundary.emplace_back(center - root_half_width, addendum + y_shift);
    if (radius <= 1e-12) {
      boundary.emplace_back(
          center - sharp_tip_half_width,
          -dedendum + y_shift);
      boundary.emplace_back(
          center + sharp_tip_half_width,
          -dedendum + y_shift);
    } else {
      const double transition =
          radius * (1.0 - std::sin(alpha)) / std::cos(alpha);
      const Complex left_center(
          center - sharp_tip_half_width + transition,
          -dedendum + radius + y_shift);
      for (int index = 0; index <= 6; ++index) {
        const double angle = std::lerp(
            kPi + alpha,
            1.5 * kPi,
            static_cast<double>(index) / 6.0);
        boundary.push_back(left_center + radius * cis(angle));
      }
      const Complex right_center(
          center + sharp_tip_half_width - transition,
          -dedendum + radius + y_shift);
      boundary.emplace_back(right_center.real(), -dedendum + y_shift);
      for (int index = 0; index <= 6; ++index) {
        const double angle = std::lerp(
            -0.5 * kPi,
            -alpha,
            static_cast<double>(index) / 6.0);
        boundary.push_back(right_center + radius * cis(angle));
      }
    }
    boundary.emplace_back(center + root_half_width, addendum + y_shift);
    boundary.emplace_back(center + 0.5 * pitch, addendum + y_shift);
  }

  void append_cycloidal_tooth(
      std::vector<Complex>& boundary,
      double center,
      double margin) const {
    const double pitch = kPi * config.module;
    const double root_half_width =
        0.25 * pitch + addendum * std::tan(alpha);
    const double tip_half_width =
        0.25 * pitch - dedendum * std::tan(alpha);
    const double blend = std::clamp(config.cycloidal_rolling_factor, 0.0, 1.0);
    const double y_shift = -margin;

    boundary.emplace_back(center - 0.5 * pitch, addendum + y_shift);
    boundary.emplace_back(center - root_half_width, addendum + y_shift);
    for (int index = 1; index <= 14; ++index) {
      const double q = static_cast<double>(index) / 14.0;
      const double tau = kPi * q;
      const double cycloid_x = (tau - std::sin(tau)) / kPi;
      const double cycloid_y = 0.5 * (1.0 - std::cos(tau));
      const double x_fraction = std::lerp(q, cycloid_x, blend);
      const double y_fraction = std::lerp(q, cycloid_y, blend);
      boundary.emplace_back(
          center - std::lerp(root_half_width, tip_half_width, x_fraction),
          std::lerp(addendum, -dedendum, y_fraction) + y_shift);
    }
    boundary.emplace_back(center + tip_half_width, -dedendum + y_shift);
    for (int index = 13; index >= 0; --index) {
      const double q = static_cast<double>(index) / 14.0;
      const double tau = kPi * q;
      const double cycloid_x = (tau - std::sin(tau)) / kPi;
      const double cycloid_y = 0.5 * (1.0 - std::cos(tau));
      const double x_fraction = std::lerp(q, cycloid_x, blend);
      const double y_fraction = std::lerp(q, cycloid_y, blend);
      boundary.emplace_back(
          center + std::lerp(root_half_width, tip_half_width, x_fraction),
          std::lerp(addendum, -dedendum, y_fraction) + y_shift);
    }
    boundary.emplace_back(center + 0.5 * pitch, addendum + y_shift);
  }

  std::vector<Complex> make_rack(
      double common_arc,
      double phase_offset,
      double half_width,
      double top,
      double margin) const {
    const double pitch = kPi * config.module;
    const double first_center =
        phase_offset - common_arc +
        std::floor((-half_width - phase_offset + common_arc) / pitch) * pitch;
    std::vector<Complex> boundary;
    boundary.reserve(
        static_cast<std::size_t>(
            std::ceil(2.0 * half_width / pitch) * 20.0 + 8.0));
    boundary.emplace_back(first_center - pitch, addendum - margin);
    for (double center = first_center - 0.5 * pitch;
         center <= half_width + pitch;
         center += pitch) {
      if (config.profile_family == ProfileFamily::kCycloidalRack) {
        append_cycloidal_tooth(boundary, center, margin);
      } else {
        append_involute_tooth(boundary, center, margin);
      }
    }
    const double right = boundary.back().real();
    const double left = boundary.front().real();
    boundary.emplace_back(right, top);
    boundary.emplace_back(left, top);
    return boundary;
  }

  std::vector<Point> generate_swept_gear(
      bool driven,
      int samples_per_radian,
      int& phase_count_out) {
    const double open_padding =
        2.5 * (active_end - active_start) / static_cast<double>(drive_teeth);
    const double cycle_start =
        is_closed() ? active_start
                    : std::max(domain_start(), active_start - open_padding);
    const double cycle_end =
        is_closed()
            ? active_start + (driven ? driven_cycle : drive_cycle)
            : std::min(domain_end(), active_end + open_padding);
    const int teeth = driven ? driven_teeth : drive_teeth;
    const double span = cycle_end - cycle_start;
    const int phases_per_tooth = 24;
    const int phase_count = std::max(
        phases_per_tooth * teeth,
        static_cast<int>(
            std::ceil(std::abs(span) * static_cast<double>(samples_per_radian))));
    phase_count_out = phase_count;

    const double pitch = kPi * config.module;
    const double blank_radius =
        maximum_pitch_radius + addendum + 2.5 * config.module;
    const double rack_half_width = 2.4 * blank_radius + 2.0 * pitch;
    const double rack_top = 2.5 * blank_radius + pitch;
    const double sweep_margin = 2e-4 * config.module;

    ExactPolygonSet gear;
    gear.insert(make_circle_polygon(blank_radius, 1024));
    std::vector<ExactPolygon> cutters;
    cutters.reserve(static_cast<std::size_t>(phase_count));
    for (int index = 0; index < phase_count; ++index) {
      const double fraction =
          static_cast<double>(index) / static_cast<double>(phase_count);
      const double phi = std::lerp(cycle_start, cycle_end, fraction);
      const double common_arc =
          center_distance * integral.integral(active_start, phi);
      const Complex pitch_point =
          driven ? driven_centrode(phi) : drive_centrode(phi);
      const Complex tangent =
          driven ? driven_tangent(phi) : drive_tangent(phi);
      const Complex outward_normal =
          driven ? Complex(0.0, -1.0) * tangent
                 : Complex(0.0, 1.0) * tangent;
      const double phase_offset = driven ? 0.0 : 0.5 * pitch;
      const std::vector<Complex> rack = make_rack(
          common_arc,
          phase_offset,
          rack_half_width,
          rack_top,
          sweep_margin);
      const ExactPolygon cutter = transform_rack_polygon(
          rack, pitch_point, tangent, outward_normal);
      if (!cutter.is_simple()) {
        throw std::runtime_error("Generated rack cutter pose is not simple.");
      }
      cutters.push_back(cutter);
    }
    // CGAL's range join uses an aggregate divide-and-conquer overlay. This
    // avoids overlaying every cutter against an increasingly complex gear.
    ExactPolygonSet swept_cutters;
    swept_cutters.join(cutters.begin(), cutters.end());
    gear.difference(swept_cutters);

    if (!is_closed()) {
      double start_angle = 0.0;
      double end_angle = 0.0;
      if (driven) {
        start_angle = psi(config.transmission.active_start) + kPi;
        end_angle = psi(config.transmission.active_end) + kPi;
      } else {
        start_angle = -config.transmission.active_start;
        end_angle = -config.transmission.active_end;
      }
      if (std::abs(end_angle - start_angle) >= 2.0 * kPi - 1e-8) {
        throw std::invalid_argument(
            "Open gear body span must be less than one body revolution.");
      }
      const double inner_radius =
          std::max(0.08 * config.module, 0.22 * minimum_pitch_radius);
      gear.intersection(
          make_open_sector(start_angle, end_angle, inner_radius, blank_radius));
    }

    std::list<ExactPolygonWithHoles> components;
    gear.polygons_with_holes(std::back_inserter(components));
    if (components.empty()) {
      throw std::runtime_error("Cutter sweep removed the entire gear blank.");
    }
    const auto selected = std::max_element(
        components.begin(),
        components.end(),
        [](const ExactPolygonWithHoles& lhs, const ExactPolygonWithHoles& rhs) {
          return exact_component_area(lhs) < exact_component_area(rhs);
        });
    if (selected->outer_boundary().size() < 3) {
      throw std::runtime_error("Cutter sweep produced an empty outer boundary.");
    }
    return simplify_outline(
        selected->outer_boundary(),
        2e-7 * std::max(1.0, config.module));
  }

  std::vector<Point> generate_conjugate_mate(
      const std::vector<Point>& master_outline,
      int samples_per_radian,
      int& phase_count_out) {
    const double open_padding =
        2.5 * (active_end - active_start) / static_cast<double>(drive_teeth);
    const double cycle_start =
        is_closed() ? active_start
                    : std::max(domain_start(), active_start - open_padding);
    const double cycle_end =
        is_closed() ? active_start + driven_cycle
                    : std::min(domain_end(), active_end + open_padding);
    const double span = cycle_end - cycle_start;
    const int phase_count = std::max(
        24 * driven_teeth,
        static_cast<int>(
            std::ceil(std::abs(span) *
                      static_cast<double>(samples_per_radian))));
    phase_count_out = phase_count;

    const double blank_radius =
        maximum_pitch_radius + addendum + 2.5 * config.module;
    ExactPolygonSet gear;
    gear.insert(make_circle_polygon(blank_radius, 1024));
    const std::vector<Point> master = simplify_closed_for_sweep(
        master_outline, 2e-4 * std::max(1.0, config.module));
    std::vector<ExactPolygon> cutters;
    cutters.reserve(static_cast<std::size_t>(phase_count));

    for (int index = 0; index < phase_count; ++index) {
      const double fraction =
          static_cast<double>(index) / static_cast<double>(phase_count);
      const double phi = std::lerp(cycle_start, cycle_end, fraction);
      const double drive_delta = phi - active_start;
      const double driven_delta = psi(phi) - psi(active_start);
      const Complex rotation = cis(drive_delta + driven_delta);
      const Complex translation =
          -center_distance * cis(driven_delta);
      ExactPolygon cutter;
      for (std::size_t point_index = 0;
           point_index + 1 < master.size();
           ++point_index) {
        const Complex point(master[point_index].x(), master[point_index].y());
        cutter.push_back(to_exact_point(translation + rotation * point));
      }
      if (cutter.orientation() == CGAL::CLOCKWISE) {
        cutter.reverse_orientation();
      }
      if (!cutter.is_simple()) {
        throw std::runtime_error(
            "Simplified master gear is not a valid conjugate cutter.");
      }
      cutters.push_back(cutter);
    }
    ExactPolygonSet swept_cutters;
    swept_cutters.join(cutters.begin(), cutters.end());
    gear.difference(swept_cutters);

    if (!is_closed()) {
      const double start_angle =
          psi(config.transmission.active_start) + kPi;
      const double end_angle =
          psi(config.transmission.active_end) + kPi;
      const double inner_radius =
          std::max(0.08 * config.module, 0.22 * minimum_pitch_radius);
      gear.intersection(
          make_open_sector(start_angle, end_angle, inner_radius, blank_radius));
    }

    std::list<ExactPolygonWithHoles> components;
    gear.polygons_with_holes(std::back_inserter(components));
    if (components.empty()) {
      throw std::runtime_error(
          "Conjugate sweep removed the entire mate gear blank.");
    }
    const auto selected = std::max_element(
        components.begin(),
        components.end(),
        [](const ExactPolygonWithHoles& lhs,
           const ExactPolygonWithHoles& rhs) {
          return exact_component_area(lhs) < exact_component_area(rhs);
        });
    return simplify_outline(
        selected->outer_boundary(),
        2e-7 * std::max(1.0, config.module));
  }

  void verify_pair(
      const std::vector<Point>& drive,
      const std::vector<Point>& driven) {
    placed_pair_overlap = 0.0;
    verification_phase_count =
        is_closed() ? std::max(64, 4 * std::max(drive_teeth, driven_teeth)) : 48;
    const std::vector<double> phase_overlaps = parallel_values(
        verification_phase_count,
        [&](int index) {
          const double fraction =
              static_cast<double>(index) /
              static_cast<double>(
                  verification_phase_count - (is_closed() ? 0 : 1));
          const double phi =
              std::lerp(active_start, active_end, fraction);
          return placed_pair_overlap_area(
              drive,
              driven,
              center_distance,
              phi - active_start,
              -(psi(phi) - psi(active_start)));
        });
    placed_pair_overlap =
        *std::max_element(phase_overlaps.begin(), phase_overlaps.end());
    const double tolerance =
        1e-6 * config.module * config.module *
        static_cast<double>(std::max(drive_teeth, driven_teeth));
    if (placed_pair_overlap > tolerance) {
      std::ostringstream message;
      message << "Continuous-phase verification found solid overlap area "
              << placed_pair_overlap << " above tolerance " << tolerance << '.';
      throw std::runtime_error(message.str());
    }

    const double contact_area_tolerance =
        1e-11 * config.module * config.module;
    maximum_transmission_error = 0.0;
    const int recovery_phase_count = is_closed() ? 6 : 4;
    const std::vector<double> contact_deltas = parallel_values(
        recovery_phase_count,
        [&](int index) {
          const double fraction =
              (static_cast<double>(index) + 0.37) /
              static_cast<double>(recovery_phase_count);
          const double phi =
              std::lerp(active_start, active_end, fraction);
          const double drive_angle = phi - active_start;
          const double desired_driven_angle =
              -(psi(phi) - psi(active_start));
          ExactPolygonSet placed_drive;
          placed_drive.insert(
              transform_polygon(drive, drive_angle, 0.0));
          if (placed_pair_overlap_area(
                  placed_drive,
                  driven,
                  center_distance,
                  desired_driven_angle) > contact_area_tolerance) {
            return 0.0;
          }

          double best_contact_delta =
              std::numeric_limits<double>::infinity();
          for (const double direction : {-1.0, 1.0}) {
            double clear_delta = 0.0;
            double collision_delta = 1e-5;
            bool found = false;
            while (collision_delta <= 0.08) {
              const double overlap = placed_pair_overlap_area(
                  placed_drive,
                  driven,
                  center_distance,
                  desired_driven_angle + direction * collision_delta);
              if (overlap > contact_area_tolerance) {
                found = true;
                break;
              }
              clear_delta = collision_delta;
              collision_delta *= 2.0;
            }
            if (!found) {
              continue;
            }
            for (int iteration = 0; iteration < 16; ++iteration) {
              const double midpoint =
                  0.5 * (clear_delta + collision_delta);
              const double overlap = placed_pair_overlap_area(
                  placed_drive,
                  driven,
                  center_distance,
                  desired_driven_angle + direction * midpoint);
              if (overlap > contact_area_tolerance) {
                collision_delta = midpoint;
              } else {
                clear_delta = midpoint;
              }
            }
            best_contact_delta =
                std::min(best_contact_delta, collision_delta);
          }
          if (!std::isfinite(best_contact_delta)) {
            throw std::runtime_error(
                "Finished solids do not establish positive contact near the requested motion.");
          }
          return best_contact_delta;
        });
    maximum_transmission_error =
        *std::max_element(contact_deltas.begin(), contact_deltas.end());
  }

  GenerationResult run(int samples_per_radian) {
    if (samples_per_radian < 20) {
      throw std::invalid_argument("samples_per_radian must be at least 20.");
    }
    measure_centrodes();
    int drive_phases = 0;
    int driven_phases = 0;
    std::vector<Point> drive;
    std::vector<Point> driven;
    if (config.profile_family == ProfileFamily::kCycloidalRack) {
      drive =
          generate_swept_gear(false, samples_per_radian, drive_phases);
      driven = generate_conjugate_mate(
          drive, samples_per_radian, driven_phases);
    } else {
      std::future<std::vector<Point>> driven_generation;
      try {
        driven_generation = std::async(
            std::launch::async,
            [this, samples_per_radian, &driven_phases]() {
              return generate_swept_gear(
                  true, samples_per_radian, driven_phases);
            });
      } catch (const std::system_error&) {
        drive =
            generate_swept_gear(false, samples_per_radian, drive_phases);
        driven =
            generate_swept_gear(true, samples_per_radian, driven_phases);
      }
      if (driven_generation.valid()) {
        drive =
            generate_swept_gear(false, samples_per_radian, drive_phases);
        driven = driven_generation.get();
      }
    }
    if (!is_simple_closed_polygon(drive) || !is_simple_closed_polygon(driven)) {
      throw std::runtime_error(
          "Swept cutter did not produce simple hub-connected gear outlines.");
    }
    verify_pair(drive, driven);

    const auto radial_less = [](const Point& lhs, const Point& rhs) {
      return lhs.x() * lhs.x() + lhs.y() * lhs.y() <
             rhs.x() * rhs.x() + rhs.y() * rhs.y();
    };
    const Point& drive_root =
        *std::min_element(drive.begin(), drive.end() - 1, radial_less);
    const Point& driven_root =
        *std::min_element(driven.begin(), driven.end() - 1, radial_less);
    const double minimum_root_value = std::min(
        std::hypot(drive_root.x(), drive_root.y()),
        std::hypot(driven_root.x(), driven_root.y()));
    double maximum_ratio = 0.0;
    for (int index = 0; index <= 4096; ++index) {
      const double phi = std::lerp(
          active_start,
          active_end,
          static_cast<double>(index) / 4096.0);
      maximum_ratio = std::max(maximum_ratio, psi1(phi));
    }
    const double contact_distance_bound =
        addendum / std::max(1e-6, std::sin(alpha)) +
        0.5 * kPi * config.module;

    return {
        config,
        drive_teeth,
        driven_teeth,
        average_angular_ratio,
        total_integral,
        center_distance,
        curvature_limit,
        0.0,
        0.0,
        placed_pair_overlap,
        centrodes_are_convex,
        maximum_drive_curvature,
        minimum_driven_curvature,
        drive_phases + driven_phases,
        verification_phase_count,
        std::max(drive_cycle / static_cast<double>(drive_phases),
                 driven_cycle / static_cast<double>(driven_phases)),
        maximum_transmission_error,
        (1.0 + maximum_ratio) * contact_distance_bound,
        minimum_root_value,
        0.5 * kPi * config.module - 2.0 * addendum * std::tan(alpha),
        {},
        std::move(drive),
        std::move(driven),
    };
  }

  SampleConfig config;
  MotionLaw motion;
  double alpha = 0.0;
  double addendum = 0.0;
  double dedendum = 0.0;
  double fillet_radius = 0.0;
  IntegralTable integral;
  int drive_teeth = 0;
  int driven_teeth = 0;
  double active_start = 0.0;
  double active_end = 0.0;
  double drive_cycle = 0.0;
  double driven_cycle = 0.0;
  double average_angular_ratio = 1.0;
  double total_integral = 0.0;
  double center_distance = 0.0;
  double curvature_limit = 0.0;
  bool centrodes_are_convex = true;
  double maximum_drive_curvature = 0.0;
  double minimum_driven_curvature = 0.0;
  double maximum_pitch_radius = 0.0;
  double minimum_pitch_radius = 0.0;
  double placed_pair_overlap = 0.0;
  double maximum_transmission_error = 0.0;
  int verification_phase_count = 0;
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
          false,
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
          false,
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
          false,
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
          false,
      },
      {
          "nonconvex_inflected",
          "Nonconvex regression: psi(phi) = phi + 0.18 sin(2 phi)",
          {{2, 0.18}},
          32,
          1.0,
          20.0,
          1.0,
          1.2,
          0.3,
          GearTopology::kClosed,
          {},
          true,
      },
      {
          "cycloidal_two_lobe",
          "Cycloidal-rack pair: psi(phi) = phi - 0.08 sin(2 phi)",
          {{2, -0.08}},
          28,
          1.0,
          20.0,
          1.0,
          1.15,
          0.2,
          GearTopology::kClosed,
          {},
          false,
          ProfileFamily::kCycloidalRack,
          0.35,
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
  return Impl(config_).run(samples_per_radian);
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

void write_result(
    const GenerationResult& result,
    const std::filesystem::path& directory) {
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
           << "  \"description\": \"" << json_escape(result.config.description)
           << "\",\n"
           << "  \"topology\": \""
           << (result.config.topology == GearTopology::kClosed ? "closed" : "open")
           << "\",\n"
           << "  \"profile_family\": \""
           << (result.config.profile_family == ProfileFamily::kCycloidalRack
                   ? "cycloidal_rack"
                   : "involute_rack")
           << "\",\n"
           << "  \"drive_teeth\": " << result.drive_teeth << ",\n"
           << "  \"driven_teeth\": " << result.driven_teeth << ",\n"
           << "  \"average_angular_ratio\": "
           << result.average_angular_ratio << ",\n"
           << "  \"module\": " << result.config.module << ",\n"
           << "  \"pressure_angle_deg\": "
           << result.config.pressure_angle_deg << ",\n"
           << "  \"total_integral\": " << result.total_integral << ",\n"
           << "  \"center_distance\": " << result.center_distance << ",\n"
           << "  \"undercut_curvature_limit\": "
           << result.undercut_curvature_limit << ",\n"
           << "  \"maximum_join_gap\": " << result.maximum_join_gap << ",\n"
           << "  \"maximum_intersection_residual\": "
           << result.maximum_intersection_residual << ",\n"
           << "  \"placed_pair_overlap_area\": "
           << result.placed_pair_overlap_area << ",\n"
           << "  \"centrodes_are_convex\": "
           << (result.centrodes_are_convex ? "true" : "false") << ",\n"
           << "  \"maximum_drive_curvature\": "
           << result.maximum_drive_curvature << ",\n"
           << "  \"minimum_driven_curvature\": "
           << result.minimum_driven_curvature << ",\n"
           << "  \"cutter_sweep_phase_count\": "
           << result.cutter_sweep_phase_count << ",\n"
           << "  \"verification_phase_count\": "
           << result.verification_phase_count << ",\n"
           << "  \"sweep_angular_step\": " << result.sweep_angular_step << ",\n"
           << "  \"maximum_transmission_error\": "
           << result.maximum_transmission_error << ",\n"
           << "  \"maximum_sliding_velocity_factor\": "
           << result.maximum_sliding_velocity_factor << ",\n"
           << "  \"minimum_root_radius\": "
           << result.minimum_root_radius << ",\n"
           << "  \"minimum_tip_thickness\": "
           << result.minimum_tip_thickness << ",\n"
           << "  \"drive_area\": "
           << signed_polygon_area(result.drive_outline) << ",\n"
           << "  \"driven_area\": "
           << signed_polygon_area(result.driven_outline) << "\n"
           << "}\n";
}

}  // namespace ncgear
