#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <complex>
#include <filesystem>
#include <string>
#include <vector>

namespace ncgear {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_2;

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Harmonic {
  int order = 1;
  double amplitude = 0.0;
};

enum class GearTopology {
  kClosed,
  kOpen,
};

struct TransmissionSamples {
  double domain_start = 0.0;
  double domain_end = 2.0 * kPi;
  double active_start = 0.0;
  double active_end = 2.0 * kPi;
  double period = 2.0 * kPi;
  double cycle_delta = 2.0 * kPi;
  std::vector<double> psi;
  std::vector<double> psi1;
  std::vector<double> psi2;
  std::vector<double> psi3;
};

struct SampleConfig {
  std::string name;
  std::string description;
  std::vector<Harmonic> harmonics;
  int teeth = 14;
  double module = 2.0;
  double pressure_angle_deg = 20.0;
  double addendum_factor = 1.0;
  double dedendum_factor = 1.2;
  double fillet_factor = 0.3;
  GearTopology topology = GearTopology::kClosed;
  TransmissionSamples transmission;
  bool allow_nonconvex_centrodes = false;
};

struct ToothCheckpoint {
  double chi = 0.0;
  double drive_singular_minus = 0.0;
  double drive_singular_plus = 0.0;
  double drive_kappa_minus = 0.0;
  double drive_kappa_plus = 0.0;
  bool drive_undercut_minus = false;
  bool drive_undercut_plus = false;
};

struct GenerationResult {
  SampleConfig config;
  int drive_teeth = 0;
  int driven_teeth = 0;
  double average_angular_ratio = 1.0;
  double total_integral = 0.0;
  double center_distance = 0.0;
  double undercut_curvature_limit = 0.0;
  double maximum_join_gap = 0.0;
  double maximum_intersection_residual = 0.0;
  double placed_pair_overlap_area = 0.0;
  bool centrodes_are_convex = true;
  double maximum_drive_curvature = 0.0;
  double minimum_driven_curvature = 0.0;
  std::vector<ToothCheckpoint> checkpoints;
  std::vector<Point> drive_outline;
  std::vector<Point> driven_outline;
};

std::vector<SampleConfig> builtin_samples();
SampleConfig builtin_sample(const std::string& name);

class GearGenerator {
 public:
  explicit GearGenerator(SampleConfig config);

  GenerationResult generate(int samples_per_radian = 110);

 private:
  struct Impl;
  SampleConfig config_;
};

bool is_simple_closed_polygon(const std::vector<Point>& points);
double signed_polygon_area(const std::vector<Point>& points);
void write_result(const GenerationResult& result, const std::filesystem::path& directory);

}  // namespace ncgear
