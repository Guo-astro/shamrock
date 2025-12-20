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
 * @file ReconstructConfig.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Configuration for reconstruction methods in GSPH
 *
 * This file contains the configuration structures for spatial reconstruction
 * methods used in Godunov SPH (GSPH). Reconstruction extrapolates primitive
 * variables to particle interfaces for higher-order accuracy.
 *
 * Based on Inutsuka (2002) "Reformulation of Smoothed Particle Hydrodynamics
 * with Riemann Solver":
 * - First-order: Piecewise constant (all gradients set to zero)
 * - Second-order: Uses gradients with monotonicity constraint on velocity
 *   (gradients set to zero at shock fronts when Mach > threshold)
 */

#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::gsph {

    /**
     * @brief Configuration for reconstruction methods in GSPH
     *
     * This struct contains the configuration for different reconstruction types:
     * - FirstOrder: Piecewise constant, no reconstruction (most diffusive)
     * - SecondOrder: Uses gradients with Inutsuka's monotonicity constraint
     *
     * Reference: Inutsuka (2002) Section 3.3
     *
     * @tparam Tvec type of the vector of coordinates
     */
    template<class Tvec>
    struct ReconstructConfig;

} // namespace shammodels::gsph

template<class Tvec>
struct shammodels::gsph::ReconstructConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    /**
     * @brief First-order reconstruction (piecewise constant)
     *
     * Sets all gradients to zero. Simple and robust but diffusive.
     * Equivalent to standard first-order Godunov method.
     * Use this for very strong shocks or debugging.
     */
    struct FirstOrder {};

    /**
     * @brief Second-order reconstruction (Inutsuka 2002)
     *
     * Uses computed gradients with monotonicity constraint.
     * At shock fronts (Mach > mach_threshold), velocity gradients
     * are set to zero to maintain stability.
     *
     * Reference: Inutsuka (2002) Eq. (59)-(61)
     */
    struct SecondOrder {
        /// Mach number threshold for monotonicity constraint (default: 1.1)
        /// When relative velocity > c_s / mach_threshold, use first-order
        Tscal mach_threshold = Tscal{1.1};
    };

    using Variant = std::variant<FirstOrder, SecondOrder>;

    Variant config = SecondOrder{};

    void set(Variant v) { config = v; }

    void set_first_order() { set(FirstOrder{}); }

    void set_second_order(Tscal mach_threshold = Tscal{1.1}) { set(SecondOrder{mach_threshold}); }

    inline bool is_first_order() const { return std::holds_alternative<FirstOrder>(config); }

    inline bool is_second_order() const { return std::holds_alternative<SecondOrder>(config); }

    inline bool requires_gradients() const { return is_second_order(); }

    inline void print_status() const {
        logger::raw_ln("--- Reconstruction config");

        if (std::get_if<FirstOrder>(&config)) {
            logger::raw_ln("  Type : FirstOrder (piecewise constant)");
        } else if (const SecondOrder *v = std::get_if<SecondOrder>(&config)) {
            logger::raw_ln("  Type           : SecondOrder (Inutsuka 2002)");
            logger::raw_ln("  mach_threshold =", v->mach_threshold);
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("-------------");
    }
};

namespace shammodels::gsph {

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const ReconstructConfig<Tvec> &p) {
        using T           = ReconstructConfig<Tvec>;
        using FirstOrder  = typename T::FirstOrder;
        using SecondOrder = typename T::SecondOrder;

        if (std::get_if<FirstOrder>(&p.config)) {
            j = {
                {"reconstruct_type", "first_order"},
            };
        } else if (const SecondOrder *v = std::get_if<SecondOrder>(&p.config)) {
            j = {
                {"reconstruct_type", "second_order"},
                {"mach_threshold", v->mach_threshold},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, ReconstructConfig<Tvec> &p) {
        using T           = ReconstructConfig<Tvec>;
        using Tscal       = shambase::VecComponent<Tvec>;
        using FirstOrder  = typename T::FirstOrder;
        using SecondOrder = typename T::SecondOrder;

        if (!j.contains("reconstruct_type")) {
            shambase::throw_with_loc<std::runtime_error>(
                "no field reconstruct_type is found in this json");
        }

        std::string reconstruct_type;
        j.at("reconstruct_type").get_to(reconstruct_type);

        if (reconstruct_type == "first_order") {
            p.set(FirstOrder{});
        } else if (reconstruct_type == "second_order") {
            Tscal mach_threshold = Tscal{1.1};
            if (j.contains("mach_threshold")) {
                j.at("mach_threshold").get_to(mach_threshold);
            }
            p.set(SecondOrder{mach_threshold});
        } else {
            shambase::throw_unimplemented("Unknown reconstruction type: " + reconstruct_type);
        }
    }

} // namespace shammodels::gsph
