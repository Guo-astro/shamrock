// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file RiemannSolvers.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Riemann solvers for GSPH
 *
 * Implements 1D Riemann solvers for computing interface states (p*, v*)
 * in Godunov SPH. These are used to compute pressure forces between
 * particle pairs.
 *
 * References:
 * - Inutsuka (2002) "Reformulation of SPH with Riemann Solver"
 * - Cha & Whitworth (2003) "Implementations and tests of Godunov-type
 *   particle hydrodynamics"
 * - Toro (2009) "Riemann Solvers and Numerical Methods for Fluid Dynamics"
 */

#include "shambackends/math.hpp"
#include <tuple>

namespace shammodels::gsph {

/**
 * @brief Iterative Riemann solver (van Leer 1997)
 *
 * Solves the 1D Riemann problem using Newton-Raphson iteration.
 * Returns the interface pressure p* and velocity v*.
 *
 * @tparam Tscal Scalar type (f32 or f64)
 * @param rho_L Left density
 * @param u_L Left velocity (projected onto pair axis)
 * @param P_L Left pressure
 * @param rho_R Right density
 * @param u_R Right velocity (projected onto pair axis)
 * @param P_R Right pressure
 * @param gamma Adiabatic index
 * @param tol Convergence tolerance
 * @param max_iter Maximum iterations
 * @return std::tuple<Tscal, Tscal> {p_star, v_star}
 */
template<class Tscal>
inline std::tuple<Tscal, Tscal> riemann_iterative(
    Tscal rho_L,
    Tscal u_L,
    Tscal P_L,
    Tscal rho_R,
    Tscal u_R,
    Tscal P_R,
    Tscal gamma,
    Tscal tol,
    u32 max_iter) {

    // Sound speeds
    const Tscal eps   = Tscal{1e-30}; // Small epsilon to prevent division by zero
    const Tscal c_L   = sycl::sqrt(gamma * sycl::max(P_L, eps) / sycl::max(rho_L, eps));
    const Tscal c_R   = sycl::sqrt(gamma * sycl::max(P_R, eps) / sycl::max(rho_R, eps));
    const Tscal rho_c = Tscal{0.5} * (rho_L * c_L + rho_R * c_R);

    // Initial guess: acoustic approximation
    Tscal p_star = sycl::max(
        Tscal{0}, Tscal{0.5} * (P_L + P_R) - Tscal{0.5} * (u_R - u_L) * rho_c);

    // Helper functions for pressure-velocity relation
    auto f_and_df = [gamma, eps](Tscal p, Tscal rho, Tscal P, Tscal c) {
        const Tscal gm1   = gamma - Tscal{1};
        const Tscal gp1   = gamma + Tscal{1};
        const Tscal ratio = p / sycl::max(P, eps);

        Tscal f, df;
        if (p > P) {
            // Shock wave
            const Tscal A = Tscal{2} / (gp1 * sycl::max(rho, eps));
            const Tscal B = gm1 / gp1 * P;
            const Tscal sq = sycl::sqrt(A / (p + B + eps));
            f  = (p - P) * sq;
            df = sq * (Tscal{1} - (p - P) / (Tscal{2} * (p + B + eps)));
        } else {
            // Rarefaction wave
            const Tscal power = gm1 / (Tscal{2} * gamma);
            f  = Tscal{2} * c / gm1 * (sycl::pow(ratio, power) - Tscal{1});
            df = sycl::pow(ratio, -gp1 / (Tscal{2} * gamma)) / (rho * c + eps);
        }
        return std::make_tuple(f, df);
    };

    // Newton-Raphson iteration
    for (u32 iter = 0; iter < max_iter; ++iter) {
        auto [f_L, df_L] = f_and_df(p_star, rho_L, P_L, c_L);
        auto [f_R, df_R] = f_and_df(p_star, rho_R, P_R, c_R);

        const Tscal f  = f_L + f_R + (u_R - u_L);
        const Tscal df = df_L + df_R;

        const Tscal dp = -f / (df + eps);
        p_star = sycl::max(tol, p_star + dp);

        if (sycl::fabs(dp) < tol * sycl::max(Tscal{1}, p_star)) {
            break;
        }
    }

    // Compute v_star from p_star
    auto [f_L, df_L] = f_and_df(p_star, rho_L, P_L, c_L);
    auto [f_R, df_R] = f_and_df(p_star, rho_R, P_R, c_R);
    const Tscal v_star = Tscal{0.5} * (u_L + u_R) + Tscal{0.5} * (f_R - f_L);

    return {p_star, v_star};
}

/**
 * @brief HLLC approximate Riemann solver
 *
 * Harten-Lax-van Leer-Contact solver. Efficient approximate solver
 * that captures contact discontinuities.
 *
 * Reference: Toro, Spruce & Speares (1994)
 *
 * @tparam Tscal Scalar type
 * @param rho_L Left density
 * @param u_L Left velocity
 * @param P_L Left pressure
 * @param c_L Left sound speed
 * @param rho_R Right density
 * @param u_R Right velocity
 * @param P_R Right pressure
 * @param c_R Right sound speed
 * @return std::tuple<Tscal, Tscal> {p_star, v_star}
 */
template<class Tscal>
inline std::tuple<Tscal, Tscal> riemann_hllc(
    Tscal rho_L,
    Tscal u_L,
    Tscal P_L,
    Tscal c_L,
    Tscal rho_R,
    Tscal u_R,
    Tscal P_R,
    Tscal c_R) {

    const Tscal eps = Tscal{1e-30};

    // Pressure estimate (PVRS - Primitive Variable Riemann Solver)
    const Tscal rho_bar = Tscal{0.5} * (rho_L + rho_R);
    const Tscal c_bar   = Tscal{0.5} * (c_L + c_R);
    const Tscal p_pvrs  = Tscal{0.5} * (P_L + P_R) - Tscal{0.5} * (u_R - u_L) * rho_bar * c_bar;
    const Tscal p_star  = sycl::max(Tscal{0}, p_pvrs);

    // Wave speed estimates using simple approach (no gamma needed for estimates)
    // For rarefaction: q = 1, for shock: q > 1
    // Since we don't have gamma, use a simpler estimate
    Tscal q_L = Tscal{1};
    Tscal q_R = Tscal{1};
    if (p_star > P_L) {
        // Approximate shock relation without gamma
        q_L = sycl::sqrt(Tscal{1} + (p_star - P_L) / (Tscal{2} * rho_L * c_L * c_L + eps));
    }
    if (p_star > P_R) {
        q_R = sycl::sqrt(Tscal{1} + (p_star - P_R) / (Tscal{2} * rho_R * c_R * c_R + eps));
    }

    const Tscal S_L = u_L - c_L * q_L;
    const Tscal S_R = u_R + c_R * q_R;

    // Contact wave speed (v_star)
    const Tscal v_star = (P_R - P_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R))
                         / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + eps);

    // p_star from HLLC
    const Tscal p_hllc = Tscal{0.5}
                         * (P_L + P_R + rho_L * (S_L - u_L) * (v_star - u_L)
                            + rho_R * (S_R - u_R) * (v_star - u_R));

    return {sycl::max(Tscal{0}, p_hllc), v_star};
}

} // namespace shammodels::gsph
