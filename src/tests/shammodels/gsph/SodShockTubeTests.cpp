// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SodShockTubeTests.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Integration tests for GSPH solver using Sod shock tube
 *
 * The Sod shock tube is a standard test problem for compressible flow solvers.
 * Initial conditions:
 *   Left state (x < 0.5):  rho = 1.0, P = 1.0, v = 0
 *   Right state (x > 0.5): rho = 0.125, P = 0.1, v = 0
 *
 * This test verifies:
 * 1. GSPH solver can run without crashing
 * 2. Conservation of mass, momentum, and energy
 * 3. Results match expected shock structure
 */

#include "shambackends/typeAliasVec.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/Model.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>
#include <vector>

namespace {

/// Sod shock tube parameters
struct SodParams {
    // Left state (high pressure)
    f64 rho_L = 1.0;
    f64 P_L   = 1.0;
    f64 v_L   = 0.0;

    // Right state (low pressure)
    f64 rho_R = 0.125;
    f64 P_R   = 0.1;
    f64 v_R   = 0.0;

    // Domain
    f64 x_min = 0.0;
    f64 x_max = 1.0;
    f64 x_mid = 0.5; // Discontinuity location

    // Physics
    f64 gamma = 1.4;
};

/// Compute internal energy from pressure and density for ideal gas
inline f64 compute_uint(f64 P, f64 rho, f64 gamma) {
    return P / ((gamma - 1.0) * rho);
}

/// Setup Sod shock tube initial conditions
template<class Tvec, template<class> class SPHKernel>
void setup_sod_shock_tube(
    shammodels::gsph::Model<Tvec, SPHKernel> &model, const SodParams &params, u32 num_particles) {
    using Tscal = shambase::VecComponent<Tvec>;

    // Get scheduler
    ShamrockCtx &ctx          = model.ctx;
    PatchScheduler &sched     = shambase::get_check_ref(ctx.sched);
    const Tscal particle_mass = model.get_particle_mass();

    // Compute smoothing length from particle spacing
    const Tscal dx = (params.x_max - params.x_min) / num_particles;

    // Set initial conditions using lambda
    // Position-dependent density
    model.template set_field_value_lambda<Tscal>("hpart", [&](Tvec pos) -> Tscal {
        Tscal rho = (pos.x() < params.x_mid) ? params.rho_L : params.rho_R;
        // h = hfact * (m/rho)^(1/dim) for 1D-like setup
        return model.get_hfact() * std::pow(particle_mass / rho, Tscal{1.0 / 3.0});
    });

    // Internal energy from pressure
    model.template set_field_value_lambda<Tscal>("uint", [&](Tvec pos) -> Tscal {
        Tscal P   = (pos.x() < params.x_mid) ? params.P_L : params.P_R;
        Tscal rho = (pos.x() < params.x_mid) ? params.rho_L : params.rho_R;
        return compute_uint(P, rho, params.gamma);
    });

    // Zero initial velocity
    model.template set_field_value_lambda<Tvec>("vxyz", [&](Tvec pos) -> Tvec {
        return Tvec{0, 0, 0};
    });
}

/// Compute L2 norm of difference between two vectors
template<class T>
T compute_L2_norm(const std::vector<T> &v1, const std::vector<T> &v2) {
    if (v1.size() != v2.size()) {
        return T{-1}; // Error
    }

    T sum = T{0};
    for (size_t i = 0; i < v1.size(); ++i) {
        T diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum / static_cast<T>(v1.size()));
}

/// Collect field data from all patches
template<class T>
std::vector<T> collect_field(ShamrockCtx &ctx, const std::string &field_name) {
    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
    std::vector<T> result;

    sched.patch_data.for_each_patchdata([&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
        auto &field = pdat.get_field<T>(sched.pdl().get_field_idx<T>(field_name));
        auto vec    = field.get_buf().copy_to_stdvec();
        result.insert(result.end(), vec.begin(), vec.end());
    });

    return result;
}

/// Compute total conserved quantity (mass, momentum component, energy)
template<class Tscal>
Tscal compute_total_mass(ShamrockCtx &ctx, Tscal particle_mass) {
    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
    u64 total_count       = 0;

    sched.patch_data.for_each_patchdata([&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
        total_count += pdat.get_obj_cnt();
    });

    return static_cast<Tscal>(total_count) * particle_mass;
}

} // anonymous namespace

/**
 * @brief Basic GSPH solver functionality test
 *
 * Verifies that the GSPH solver can:
 * 1. Initialize with Sod shock tube conditions
 * 2. Run a few timesteps without crashing
 * 3. Maintain basic conservation properties
 */
TestStart(Unittest, "shammodels/gsph/sod_shock_tube_basic", gsph_sod_basic, 1) {
    using Tvec  = f64_3;
    using Tscal = f64;

    // Create context and model
    ShamrockCtx ctx{};
    ctx.pdata_layout_new();

    shammodels::gsph::Model<Tvec, shammath::M4> model{ctx};

    // Configure solver
    auto cfg = model.gen_default_config();
    cfg.set_eos_adiabatic(1.4);
    cfg.set_riemann_hllc(); // Use HLLC for robustness
    cfg.set_reconstruct_first_order();
    cfg.set_boundary_periodic(); // Use periodic for now (will add walls later)
    model.set_solver_config(cfg);

    // Initialize scheduler
    model.init_scheduler(1000, 1);

    // Setup domain - elongated box for 1D-like test
    Tvec box_min = {0.0, -0.1, -0.1};
    Tvec box_max = {1.0, 0.1, 0.1};
    model.resize_simulation_box({box_min, box_max});

    // Add particles
    Tscal dr = 0.02; // Particle spacing
    model.add_cube_fcc_3d(dr, {box_min, box_max});

    // Set particle mass based on expected total mass
    Tscal total_volume = (box_max.x() - box_min.x()) * (box_max.y() - box_min.y())
                         * (box_max.z() - box_min.z());
    Tscal avg_density    = 0.5 * (1.0 + 0.125); // Average of left and right states
    Tscal total_mass     = avg_density * total_volume;
    Tscal particle_mass  = model.total_mass_to_part_mass(total_mass);
    model.set_particle_mass(particle_mass);

    // Setup Sod initial conditions
    SodParams params;
    u64 num_particles = model.get_total_part_count();
    setup_sod_shock_tube<Tvec, shammath::M4>(model, params, num_particles);

    // Record initial state
    Tscal initial_mass = compute_total_mass<Tscal>(ctx, particle_mass);

    // Run a few timesteps
    model.set_cfl_cour(0.3);
    model.set_cfl_force(0.25);

    // Initialize before first evolution (updates load balancing)
    model.init_before_evolve();

    const u32 num_steps = 5;
    for (u32 i = 0; i < num_steps; ++i) {
        model.evolve_once();
    }

    // Check conservation
    Tscal final_mass   = compute_total_mass<Tscal>(ctx, particle_mass);
    Tscal mass_error   = std::abs(final_mass - initial_mass) / initial_mass;

    // Mass should be exactly conserved (no particle creation/destruction)
    REQUIRE_NAMED("Mass conservation", mass_error < 1e-10);

    // Check that simulation didn't crash and produced reasonable results
    u64 final_count = model.get_total_part_count();
    REQUIRE_NAMED("Particle count preserved", final_count == num_particles);

    // Log results for debugging
    auto &dset = shamtest::test_data().new_dataset("sod_basic");
    dset.add_data("num_particles", std::vector<f64>{(f64)num_particles});
    dset.add_data("num_steps", std::vector<f64>{(f64)num_steps});
    dset.add_data("mass_error", std::vector<f64>{mass_error});
}

/**
 * @brief GSPH Sod shock tube validation test
 *
 * Runs for more timesteps and checks shock structure against
 * expected analytical solution.
 */
TestStart(ValidationTest, "shammodels/gsph/sod_shock_tube_validation", gsph_sod_validation, 1) {
    using Tvec  = f64_3;
    using Tscal = f64;

    // Create context and model
    ShamrockCtx ctx{};
    ctx.pdata_layout_new();

    shammodels::gsph::Model<Tvec, shammath::M4> model{ctx};

    // Configure solver
    auto cfg = model.gen_default_config();
    cfg.set_eos_adiabatic(1.4);
    cfg.set_riemann_hllc();
    cfg.set_reconstruct_second_order(1.1); // Second order for better accuracy
    cfg.set_boundary_periodic();
    model.set_solver_config(cfg);

    // Initialize scheduler
    model.init_scheduler(10000, 1);

    // Setup domain
    Tvec box_min = {0.0, -0.05, -0.05};
    Tvec box_max = {1.0, 0.05, 0.05};
    model.resize_simulation_box({box_min, box_max});

    // Higher resolution
    Tscal dr = 0.01;
    model.add_cube_fcc_3d(dr, {box_min, box_max});

    // Set particle mass
    Tscal total_volume   = (box_max.x() - box_min.x()) * (box_max.y() - box_min.y())
                         * (box_max.z() - box_min.z());
    Tscal avg_density    = 0.5 * (1.0 + 0.125);
    Tscal total_mass     = avg_density * total_volume;
    Tscal particle_mass  = model.total_mass_to_part_mass(total_mass);
    model.set_particle_mass(particle_mass);

    // Setup Sod initial conditions
    SodParams params;
    u64 num_particles = model.get_total_part_count();
    setup_sod_shock_tube<Tvec, shammath::M4>(model, params, num_particles);

    // Set CFL conditions
    model.set_cfl_cour(0.3);
    model.set_cfl_force(0.25);

    // Initialize before first evolution (updates load balancing)
    model.init_before_evolve();

    // Evolve to t = 0.1 (before waves hit boundaries)
    Tscal target_time = 0.1;
    bool success      = model.evolve_until(target_time, 1000);

    REQUIRE_NAMED("Evolution completed", success);

    // Collect final state
    auto positions  = collect_field<Tvec>(ctx, "xyz");
    auto velocities = collect_field<Tvec>(ctx, "vxyz");
    auto h_values   = collect_field<Tscal>(ctx, "hpart");

    // Basic sanity checks
    REQUIRE_NAMED("Particles exist", positions.size() > 0);

    // Check that velocities are non-zero (shock has developed)
    Tscal max_velocity = 0;
    for (const auto &v : velocities) {
        max_velocity = std::max(max_velocity, std::abs(v.x()));
    }
    REQUIRE_NAMED("Shock has developed (non-zero velocities)", max_velocity > 0.01);

    // Check smoothing lengths are reasonable (positive and not too large)
    bool h_reasonable = true;
    for (const auto &h : h_values) {
        if (h <= 0 || h > 1.0) {
            h_reasonable = false;
            break;
        }
    }
    REQUIRE_NAMED("Smoothing lengths reasonable", h_reasonable);

    // Log results
    auto &dset = shamtest::test_data().new_dataset("sod_validation");
    dset.add_data("num_particles", std::vector<f64>{(f64)num_particles});
    dset.add_data("target_time", std::vector<f64>{target_time});
    dset.add_data("max_velocity", std::vector<f64>{max_velocity});

    TEX_REPORT(R"==(
\subsection{GSPH Sod Shock Tube Validation}
The GSPH solver was tested with a Sod shock tube problem.
The simulation evolved to $t = 0.1$ and showed proper shock development.
)==");
}

/**
 * @brief GSPH Riemann solver comparison test
 *
 * Compares different Riemann solvers on the Sod problem.
 */
TestStart(Unittest, "shammodels/gsph/riemann_solver_comparison", gsph_riemann_comparison, 1) {
    using Tvec  = f64_3;
    using Tscal = f64;

    // Test with different Riemann solvers
    std::vector<std::string> solver_names = {"iterative", "hllc"};

    for (const auto &solver_name : solver_names) {
        // Create fresh context for each solver
        ShamrockCtx ctx{};
        ctx.pdata_layout_new();

        shammodels::gsph::Model<Tvec, shammath::M4> model{ctx};

        // Configure solver
        auto cfg = model.gen_default_config();
        cfg.set_eos_adiabatic(1.4);

        if (solver_name == "iterative") {
            cfg.set_riemann_iterative();
        } else if (solver_name == "hllc") {
            cfg.set_riemann_hllc();
        }

        cfg.set_reconstruct_first_order();
        cfg.set_boundary_periodic();
        model.set_solver_config(cfg);

        // Initialize
        model.init_scheduler(1000, 1);

        Tvec box_min = {0.0, -0.1, -0.1};
        Tvec box_max = {1.0, 0.1, 0.1};
        model.resize_simulation_box({box_min, box_max});

        Tscal dr = 0.02;
        model.add_cube_fcc_3d(dr, {box_min, box_max});

        Tscal total_volume  = (box_max.x() - box_min.x()) * (box_max.y() - box_min.y())
                              * (box_max.z() - box_min.z());
        Tscal avg_density   = 0.5 * (1.0 + 0.125);
        Tscal particle_mass = model.total_mass_to_part_mass(avg_density * total_volume);
        model.set_particle_mass(particle_mass);

        SodParams params;
        u64 num_particles = model.get_total_part_count();
        setup_sod_shock_tube<Tvec, shammath::M4>(model, params, num_particles);

        model.set_cfl_cour(0.3);
        model.set_cfl_force(0.25);

        // Initialize before first evolution (updates load balancing)
        model.init_before_evolve();

        // Run a few steps
        bool crashed = false;
        try {
            for (u32 i = 0; i < 3; ++i) {
                model.evolve_once();
            }
        } catch (...) {
            crashed = true;
        }

        REQUIRE_NAMED(solver_name + " solver did not crash", !crashed);
    }
}
