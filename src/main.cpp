#include "ncgear/gear.hpp"

#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "Usage: ncgear_generate [--sample NAME|all] [--out DIRECTORY] "
         "[--samples-per-radian N]\n"
         "       ncgear_generate --transmission-csv FILE [--open] --name NAME "
         "--teeth N --module M --domain-start A --domain-end B "
         "[--active-start A --active-end B] [--period P --cycle-delta D] "
         "[--allow-nonconvex]\n";
}

ncgear::TransmissionSamples read_transmission_csv(
    const std::filesystem::path& path,
    double domain_start,
    double domain_end) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Unable to open transmission CSV: " + path.string());
  }

  ncgear::TransmissionSamples samples;
  samples.domain_start = domain_start;
  samples.domain_end = domain_end;
  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream row(line);
    std::array<double, 5> values{};
    std::string cell;
    for (double& value : values) {
      if (!std::getline(row, cell, ',')) {
        throw std::runtime_error("Transmission CSV rows must contain five columns.");
      }
      value = std::stod(cell);
    }
    samples.psi.push_back(values[1]);
    samples.psi1.push_back(values[2]);
    samples.psi2.push_back(values[3]);
    samples.psi3.push_back(values[4]);
  }
  return samples;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string sample_name = "all";
    std::string name = "expression";
    std::string description = "Sampled transmission";
    std::filesystem::path output = "out";
    std::filesystem::path transmission_csv;
    int samples_per_radian = 110;
    int teeth = 14;
    double module = 1.0;
    double pressure_angle_deg = 20.0;
    double addendum_factor = 1.0;
    double dedendum_factor = 1.2;
    double fillet_factor = 0.3;
    double domain_start = 0.0;
    double domain_end = 2.0 * ncgear::kPi;
    double active_start = 0.0;
    double active_end = 2.0 * ncgear::kPi;
    double period = 2.0 * ncgear::kPi;
    double cycle_delta = 2.0 * ncgear::kPi;
    bool open = false;
    bool allow_nonconvex = false;

    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--sample" && i + 1 < argc) {
        sample_name = argv[++i];
      } else if (argument == "--out" && i + 1 < argc) {
        output = argv[++i];
      } else if (argument == "--transmission-csv" && i + 1 < argc) {
        transmission_csv = argv[++i];
      } else if (argument == "--name" && i + 1 < argc) {
        name = argv[++i];
      } else if (argument == "--description" && i + 1 < argc) {
        description = argv[++i];
      } else if (argument == "--teeth" && i + 1 < argc) {
        teeth = std::stoi(argv[++i]);
      } else if (argument == "--module" && i + 1 < argc) {
        module = std::stod(argv[++i]);
      } else if (argument == "--pressure-angle-deg" && i + 1 < argc) {
        pressure_angle_deg = std::stod(argv[++i]);
      } else if (argument == "--addendum-factor" && i + 1 < argc) {
        addendum_factor = std::stod(argv[++i]);
      } else if (argument == "--dedendum-factor" && i + 1 < argc) {
        dedendum_factor = std::stod(argv[++i]);
      } else if (argument == "--fillet-factor" && i + 1 < argc) {
        fillet_factor = std::stod(argv[++i]);
      } else if (argument == "--domain-start" && i + 1 < argc) {
        domain_start = std::stod(argv[++i]);
      } else if (argument == "--domain-end" && i + 1 < argc) {
        domain_end = std::stod(argv[++i]);
      } else if (argument == "--active-start" && i + 1 < argc) {
        active_start = std::stod(argv[++i]);
      } else if (argument == "--active-end" && i + 1 < argc) {
        active_end = std::stod(argv[++i]);
      } else if (argument == "--period" && i + 1 < argc) {
        period = std::stod(argv[++i]);
      } else if (argument == "--cycle-delta" && i + 1 < argc) {
        cycle_delta = std::stod(argv[++i]);
      } else if (argument == "--open") {
        open = true;
      } else if (argument == "--allow-nonconvex") {
        allow_nonconvex = true;
      } else if (argument == "--samples-per-radian" && i + 1 < argc) {
        samples_per_radian = std::stoi(argv[++i]);
      } else if (argument == "--help" || argument == "-h") {
        usage();
        return EXIT_SUCCESS;
      } else {
        usage();
        return EXIT_FAILURE;
      }
    }

    std::vector<ncgear::SampleConfig> samples;
    if (!transmission_csv.empty()) {
      ncgear::SampleConfig config;
      config.name = name;
      config.description = description;
      config.teeth = teeth;
      config.module = module;
      config.pressure_angle_deg = pressure_angle_deg;
      config.addendum_factor = addendum_factor;
      config.dedendum_factor = dedendum_factor;
      config.fillet_factor = fillet_factor;
      config.topology =
          open ? ncgear::GearTopology::kOpen : ncgear::GearTopology::kClosed;
      config.transmission =
          read_transmission_csv(transmission_csv, domain_start, domain_end);
      config.transmission.active_start = active_start;
      config.transmission.active_end = active_end;
      config.transmission.period = period;
      config.transmission.cycle_delta = cycle_delta;
      config.allow_nonconvex_centrodes = allow_nonconvex;
      samples.push_back(std::move(config));
    } else if (sample_name == "all") {
      samples = ncgear::builtin_samples();
    } else {
      samples.push_back(ncgear::builtin_sample(sample_name));
    }

    for (const auto& sample : samples) {
      std::cout << "Generating " << sample.name << "...\n";
      ncgear::GearGenerator generator(sample);
      const ncgear::GenerationResult result = generator.generate(samples_per_radian);
      ncgear::write_result(result, output / sample.name);
      std::cout << "  z1: " << result.drive_teeth << "\n"
                << "  z2: " << result.driven_teeth << "\n"
                << "  average angular ratio: " << result.average_angular_ratio << "\n"
                << "  active arc integral: " << result.total_integral << "\n"
                << "  center distance: " << result.center_distance << "\n"
                << "  drive points: " << result.drive_outline.size() << "\n"
                << "  driven points: " << result.driven_outline.size() << "\n"
                << "  max join gap: " << result.maximum_join_gap << "\n"
                << "  max refined intersection residual: "
                << result.maximum_intersection_residual << "\n"
                << "  placed pair overlap area: "
                << result.placed_pair_overlap_area << "\n"
                << "  centrodes convex: "
                << (result.centrodes_are_convex ? "yes" : "no") << "\n"
                << "  max drive curvature: "
                << result.maximum_drive_curvature << "\n"
                << "  min driven curvature: "
                << result.minimum_driven_curvature << "\n";
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
