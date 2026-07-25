#include "ncgear/gear.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_near(
    double actual,
    double expected,
    double tolerance,
    const std::string& label) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(
        label + ": expected " + std::to_string(expected) + ", got " +
        std::to_string(actual));
  }
}

template <typename Function>
void require_throws_with(Function function, const std::string& text) {
  try {
    function();
  } catch (const std::exception& error) {
    require(
        std::string(error.what()).find(text) != std::string::npos,
        "unexpected error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected exception containing: " + text);
}

ncgear::TransmissionSamples sample_transmission(
    double domain_start,
    double domain_end,
    int sample_count,
    const std::function<std::array<double, 4>(double)>& evaluate,
    bool include_endpoint) {
  ncgear::TransmissionSamples samples;
  samples.domain_start = domain_start;
  samples.domain_end = domain_end;
  const double divisor =
      static_cast<double>(include_endpoint ? sample_count - 1 : sample_count);
  for (int index = 0; index < sample_count; ++index) {
    const double phi =
        std::lerp(domain_start, domain_end, static_cast<double>(index) / divisor);
    const auto values = evaluate(phi);
    samples.psi.push_back(values[0]);
    samples.psi1.push_back(values[1]);
    samples.psi2.push_back(values[2]);
    samples.psi3.push_back(values[3]);
  }
  return samples;
}

void require_valid_result(
    const ncgear::GenerationResult& result,
    const std::string& label) {
  require(
      ncgear::is_simple_closed_polygon(result.drive_outline),
      label + " drive outline is not simple");
  require(
      ncgear::is_simple_closed_polygon(result.driven_outline),
      label + " driven outline is not simple");
  require(
      std::abs(ncgear::signed_polygon_area(result.drive_outline)) > 1.0,
      label + " drive area is too small");
  require(
      std::abs(ncgear::signed_polygon_area(result.driven_outline)) > 1.0,
      label + " driven area is too small");
  require(
      result.cutter_sweep_phase_count >=
          24 * (result.drive_teeth + result.driven_teeth),
      label + " cutter sweep is under-resolved");
  require(
      result.verification_phase_count >= 48,
      label + " verification is under-resolved");
  const double overlap_limit =
      1e-6 * result.config.module * result.config.module *
      static_cast<double>(std::max(result.drive_teeth, result.driven_teeth));
  require(
      result.placed_pair_overlap_area <= overlap_limit,
      label + " pair overlap exceeds the verification tolerance");
  require(
      result.maximum_transmission_error < 0.01,
      label + " recovered contact motion differs excessively from the target");
  require(result.minimum_root_radius > 0.0, label + " has no hub-connected root");
}

void test_paper_reference() {
  const ncgear::GenerationResult result =
      ncgear::GearGenerator(ncgear::builtin_sample("paper")).generate(20);
  require_near(result.total_integral, 3.09315, 8e-6, "paper integral");
  require_near(result.center_distance, 28.4385, 8e-4, "paper center distance");
  require(result.centrodes_are_convex, "paper centrodes should be convex");
  require_valid_result(result, "paper");
}

void test_nonconvex_global_sweep() {
  ncgear::SampleConfig config = ncgear::builtin_sample("nonconvex_inflected");
  config.allow_nonconvex_centrodes = false;
  const ncgear::GenerationResult result =
      ncgear::GearGenerator(config).generate(20);
  require(!result.centrodes_are_convex, "regression centrodes should be nonconvex");
  require(
      result.maximum_drive_curvature > 0.0,
      "regression should cross a drive-centrode inflection");
  require_valid_result(result, "nonconvex");
}

void test_cycloidal_rack() {
  const ncgear::GenerationResult result =
      ncgear::GearGenerator(ncgear::builtin_sample("cycloidal_two_lobe"))
          .generate(20);
  require(
      result.config.profile_family == ncgear::ProfileFamily::kCycloidalRack,
      "cycloidal profile selection was lost");
  require_valid_result(result, "cycloidal");
}

void test_closed_sampled_unequal_ratio() {
  ncgear::SampleConfig config;
  config.name = "closed_ratio_2_to_1";
  config.description = "psi(phi) = 2 phi";
  config.teeth = 20;
  config.module = 1.0;
  config.transmission = sample_transmission(
      0.0,
      2.0 * ncgear::kPi,
      1024,
      [](double phi) {
        return std::array<double, 4>{2.0 * phi, 2.0, 0.0, 0.0};
      },
      false);
  config.transmission.period = 2.0 * ncgear::kPi;
  config.transmission.cycle_delta = 4.0 * ncgear::kPi;

  const ncgear::GenerationResult result =
      ncgear::GearGenerator(config).generate(20);
  require(result.drive_teeth == 20, "sampled drive tooth count");
  require(result.driven_teeth == 10, "sampled driven tooth count");
  require_near(result.average_angular_ratio, 2.0, 1e-10, "sampled ratio");
  require_valid_result(result, "sampled 2:1");
}

void test_open_sampled_motion() {
  constexpr double active_start = 0.0;
  constexpr double active_end = 2.4;
  constexpr double domain_start = -1.4;
  constexpr double domain_end = 3.8;
  ncgear::SampleConfig config;
  config.name = "open_variable_ratio";
  config.description = "psi(phi) = 1.8 phi + 0.03 sin(phi)";
  config.teeth = 12;
  config.module = 1.0;
  config.topology = ncgear::GearTopology::kOpen;
  config.transmission = sample_transmission(
      domain_start,
      domain_end,
      1024,
      [](double phi) {
        return std::array<double, 4>{
            1.8 * phi + 0.03 * std::sin(phi),
            1.8 + 0.03 * std::cos(phi),
            -0.03 * std::sin(phi),
            -0.03 * std::cos(phi),
        };
      },
      true);
  config.transmission.active_start = active_start;
  config.transmission.active_end = active_end;
  config.transmission.cycle_delta =
      1.8 * active_end + 0.03 * std::sin(active_end);

  const ncgear::GenerationResult result =
      ncgear::GearGenerator(config).generate(20);
  require_near(result.average_angular_ratio, 1.80844, 1e-5, "open ratio");
  require_valid_result(result, "open");
}

void test_incompatible_closed_motion() {
  ncgear::SampleConfig config;
  config.name = "incompatible_cycle";
  config.description = "psi(phi) = 2 phi - 0.04 sin(3 phi)";
  config.teeth = 30;
  config.transmission = sample_transmission(
      0.0,
      2.0 * ncgear::kPi,
      1024,
      [](double phi) {
        return std::array<double, 4>{
            2.0 * phi - 0.04 * std::sin(3.0 * phi),
            2.0 - 0.12 * std::cos(3.0 * phi),
            0.36 * std::sin(3.0 * phi),
            1.08 * std::cos(3.0 * phi),
        };
      },
      false);
  config.transmission.period = 2.0 * ncgear::kPi;
  config.transmission.cycle_delta = 4.0 * ncgear::kPi;
  require_throws_with(
      [&config]() { ncgear::GearGenerator(config).generate(20); },
      "not compatible with one driven-gear revolution");
}

void test_sampled_drive_centrode() {
  ncgear::SampleConfig config;
  config.name = "sampled_centrode";
  config.description = "r(phi) = 1 + 0.08 cos(2 phi)";
  config.teeth = 20;
  config.module = 1.0;
  constexpr int sample_count = 1024;
  for (int index = 0; index < sample_count; ++index) {
    const double phi =
        2.0 * ncgear::kPi * static_cast<double>(index) /
        static_cast<double>(sample_count);
    config.centrode.radius.push_back(1.0 + 0.08 * std::cos(2.0 * phi));
    config.centrode.radius1.push_back(-0.16 * std::sin(2.0 * phi));
    config.centrode.radius2.push_back(-0.32 * std::cos(2.0 * phi));
  }

  const ncgear::GenerationResult result =
      ncgear::GearGenerator(config).generate(20);
  require(
      !result.config.centrode.radius.empty(),
      "centrode input mode was not retained");
  require(
      result.config.centrode.reference_center_distance > 1.08,
      "centrode reference center distance was not solved");
  require_near(result.average_angular_ratio, 1.0, 1e-8, "centrode ratio");
  require(result.drive_teeth == result.driven_teeth, "centrode tooth counts");
  require_valid_result(result, "centrode");
}

void test_disconnected_deep_centrode_is_rejected() {
  ncgear::SampleConfig config;
  config.name = "disconnected_deep_five_lobe";
  config.description = "r(phi) = 1 + 0.18 cos(5 phi), ratio 5:2";
  config.teeth = 100;
  config.module = 1.0;
  config.profile_family = ncgear::ProfileFamily::kCycloidalRack;
  config.centrode.target_cycle_delta = 5.0 * ncgear::kPi;
  constexpr int sample_count = 1024;
  for (int index = 0; index < sample_count; ++index) {
    const double phi =
        2.0 * ncgear::kPi * static_cast<double>(index) /
        static_cast<double>(sample_count);
    config.centrode.radius.push_back(1.0 + 0.18 * std::cos(5.0 * phi));
    config.centrode.radius1.push_back(-0.9 * std::sin(5.0 * phi));
    config.centrode.radius2.push_back(-4.5 * std::cos(5.0 * phi));
  }

  require_throws_with(
      [&config]() { ncgear::GearGenerator(config).generate(20); },
      "Rack sweep disconnected the intended drive-gear body");
}

void test_open_sampled_drive_centrode() {
  constexpr double domain_start = -1.4;
  constexpr double active_start = 0.0;
  constexpr double active_end = 2.4;
  constexpr double domain_end = 3.8;
  constexpr int sample_count = 1024;
  ncgear::SampleConfig config;
  config.name = "open_sampled_centrode";
  config.description = "Centrode equivalent to psi1 = 1.8 + 0.03 cos(phi)";
  config.teeth = 12;
  config.module = 1.0;
  config.topology = ncgear::GearTopology::kOpen;
  config.centrode.domain_start = domain_start;
  config.centrode.domain_end = domain_end;
  config.centrode.active_start = active_start;
  config.centrode.active_end = active_end;
  config.centrode.reference_center_distance = 1.0;
  for (int index = 0; index < sample_count; ++index) {
    const double phi = std::lerp(
        domain_start,
        domain_end,
        static_cast<double>(index) /
            static_cast<double>(sample_count - 1));
    const double ratio = 1.8 + 0.03 * std::cos(phi);
    const double ratio1 = -0.03 * std::sin(phi);
    const double ratio2 = -0.03 * std::cos(phi);
    const double denominator = 1.0 + ratio;
    config.centrode.radius.push_back(ratio / denominator);
    config.centrode.radius1.push_back(
        ratio1 / (denominator * denominator));
    config.centrode.radius2.push_back(
        ratio2 / (denominator * denominator) -
        2.0 * ratio1 * ratio1 /
            (denominator * denominator * denominator));
  }

  const ncgear::GenerationResult result =
      ncgear::GearGenerator(config).generate(20);
  require_near(
      result.average_angular_ratio, 1.80844, 1e-5, "open centrode ratio");
  require_valid_result(result, "open centrode");
}

}  // namespace

int main() {
  try {
    test_paper_reference();
    test_nonconvex_global_sweep();
    test_cycloidal_rack();
    test_closed_sampled_unequal_ratio();
    test_open_sampled_motion();
    test_incompatible_closed_motion();
    test_sampled_drive_centrode();
    test_disconnected_deep_centrode_is_rejected();
    test_open_sampled_drive_centrode();
    std::cout << "All ncgear tests passed.\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
