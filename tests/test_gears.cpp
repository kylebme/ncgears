#include "ncgear/gear.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_near(double actual, double expected, double tolerance, const std::string& label) {
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
  samples.psi.reserve(static_cast<std::size_t>(sample_count));
  samples.psi1.reserve(static_cast<std::size_t>(sample_count));
  samples.psi2.reserve(static_cast<std::size_t>(sample_count));
  samples.psi3.reserve(static_cast<std::size_t>(sample_count));
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

void test_paper_checkpoints() {
  ncgear::GearGenerator generator(ncgear::builtin_sample("paper"));
  const ncgear::GenerationResult result = generator.generate(90);

  require_near(result.total_integral, 3.09315, 8e-6, "paper integral");
  require_near(result.center_distance, 28.4385, 8e-4, "paper center distance");
  require_near(
      result.undercut_curvature_limit,
      0.0583369,
      2e-7,
      "paper undercut limit");

  const std::vector<double> expected_chi = {
      0.0,
      0.674065,
      1.18877,
      1.63010,
      2.03297,
      2.41317,
      2.78037,
      ncgear::kPi,
      3.50282,
      3.87002,
      4.25022,
      4.65309,
      5.09441,
      5.60912,
  };
  require(result.checkpoints.size() == expected_chi.size(), "paper checkpoint count");
  for (std::size_t index = 0; index < expected_chi.size(); ++index) {
    require_near(
        result.checkpoints[index].chi,
        expected_chi[index],
        7e-6,
        "paper chi " + std::to_string(index + 1));
  }

  require_near(
      result.checkpoints.front().drive_singular_minus,
      -0.662309,
      8e-6,
      "paper tooth 1 minus singular");
  require_near(
      result.checkpoints.front().drive_singular_plus,
      0.662309,
      8e-6,
      "paper tooth 1 plus singular");
  require_near(
      result.checkpoints.back().drive_singular_plus,
      6.65339,
      8e-6,
      "paper tooth 14 plus singular");

  require(!result.checkpoints[1].drive_undercut_minus, "tooth 2 minus should be free");
  require(!result.checkpoints[13].drive_undercut_plus, "tooth 14 plus should be free");
  require(result.checkpoints[0].drive_undercut_minus, "tooth 1 minus should undercut");
  require(result.checkpoints[0].drive_undercut_plus, "tooth 1 plus should undercut");
}

void test_all_samples() {
  for (const ncgear::SampleConfig& sample : ncgear::builtin_samples()) {
    ncgear::GearGenerator generator(sample);
    const ncgear::GenerationResult result = generator.generate(75);
    require(ncgear::is_simple_closed_polygon(result.drive_outline), sample.name + " drive simple");
    require(
        ncgear::is_simple_closed_polygon(result.driven_outline),
        sample.name + " driven simple");
    require(
        std::abs(ncgear::signed_polygon_area(result.drive_outline)) > 1.0,
        sample.name + " drive area");
    require(
        std::abs(ncgear::signed_polygon_area(result.driven_outline)) > 1.0,
        sample.name + " driven area");
    require(result.maximum_join_gap < 2e-6, sample.name + " continuity");
    require(
        result.maximum_intersection_residual < 1e-7,
        sample.name + " intersection residual");
    require(
        result.placed_pair_overlap_area < 1e-10,
        sample.name + " placed pair overlap");
  }
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
      4096,
      [](double phi) {
        return std::array<double, 4>{2.0 * phi, 2.0, 0.0, 0.0};
      },
      false);
  config.transmission.period = 2.0 * ncgear::kPi;
  config.transmission.cycle_delta = 4.0 * ncgear::kPi;

  const ncgear::GenerationResult result = ncgear::GearGenerator(config).generate(70);
  require(result.drive_teeth == 20, "closed sampled drive tooth count");
  require(result.driven_teeth == 10, "closed sampled driven tooth count");
  require_near(result.average_angular_ratio, 2.0, 1e-10, "closed sampled ratio");
  require(result.maximum_join_gap < 2e-6, "closed sampled continuity");
  require(result.placed_pair_overlap_area < 1e-10, "closed sampled overlap");
}

void test_open_sampled_variable_ratio() {
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
      4096,
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

  const ncgear::GenerationResult result = ncgear::GearGenerator(config).generate(100);
  require(result.drive_teeth == 12, "open sampled drive tooth count");
  require(result.driven_teeth == 12, "open sampled driven tooth count");
  require_near(result.average_angular_ratio, 1.80844, 1e-5, "open sampled ratio");
  require(ncgear::is_simple_closed_polygon(result.drive_outline), "open drive simple");
  require(ncgear::is_simple_closed_polygon(result.driven_outline), "open driven simple");
  require(result.maximum_join_gap < 2e-6, "open sampled continuity");
  require(result.maximum_intersection_residual < 1e-7, "open sampled intersections");
  require(result.placed_pair_overlap_area < 2.5e-5, "open sampled overlap tolerance");
}

void test_closed_driven_cycle_validation() {
  ncgear::SampleConfig incompatible;
  incompatible.name = "incompatible_cycle";
  incompatible.description = "psi(phi) = 2 phi - 0.04 sin(3 phi)";
  incompatible.teeth = 30;
  incompatible.transmission = sample_transmission(
      0.0,
      2.0 * ncgear::kPi,
      4096,
      [](double phi) {
        return std::array<double, 4>{
            2.0 * phi - 0.04 * std::sin(3.0 * phi),
            2.0 - 0.12 * std::cos(3.0 * phi),
            0.36 * std::sin(3.0 * phi),
            1.08 * std::cos(3.0 * phi),
        };
      },
      false);
  incompatible.transmission.period = 2.0 * ncgear::kPi;
  incompatible.transmission.cycle_delta = 4.0 * ncgear::kPi;
  require_throws_with(
      [&incompatible]() { ncgear::GearGenerator(incompatible).generate(50); },
      "not periodic over one driven-gear revolution");

  ncgear::SampleConfig misaligned;
  misaligned.name = "misaligned_cycle";
  misaligned.description = "psi(phi) = 5/3 phi - 0.03 sin(5 phi)";
  misaligned.teeth = 30;
  misaligned.transmission = sample_transmission(
      0.0,
      2.0 * ncgear::kPi,
      4096,
      [](double phi) {
        return std::array<double, 4>{
            (5.0 / 3.0) * phi - 0.03 * std::sin(5.0 * phi),
            5.0 / 3.0 - 0.15 * std::cos(5.0 * phi),
            0.75 * std::sin(5.0 * phi),
            3.75 * std::cos(5.0 * phi),
        };
      },
      false);
  misaligned.transmission.period = 2.0 * ncgear::kPi;
  misaligned.transmission.cycle_delta = (10.0 / 3.0) * ncgear::kPi;
  require_throws_with(
      [&misaligned]() { ncgear::GearGenerator(misaligned).generate(50); },
      "grid does not align with the driven cycle");
}

}  // namespace

int main() {
  try {
    test_paper_checkpoints();
    test_all_samples();
    test_closed_sampled_unequal_ratio();
    test_open_sampled_variable_ratio();
    test_closed_driven_cycle_validation();
    std::cout << "All ncgear tests passed.\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
