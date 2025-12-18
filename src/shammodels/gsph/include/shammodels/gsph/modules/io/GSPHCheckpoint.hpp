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
 * @file GSPHCheckpoint.hpp
 * @author Guo (guo.yansong@optimind.tech)
 * @brief Checkpoint/restart functionality for GSPH solver
 *
 * Provides checkpoint (dump) and restart functionality for GSPH simulations.
 * Checkpoints save the complete simulation state to disk, allowing simulations
 * to be resumed from any saved point.
 *
 * File format:
 * - checkpoint_NNNN.json: Metadata (time, config, particle counts)
 * - checkpoint_NNNN.bin: Binary particle data (xyz, vxyz, hpart, uint, etc.)
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>

namespace shammodels::gsph::modules {

    /**
     * @brief Checkpoint handler for GSPH simulations
     *
     * Handles saving and loading simulation checkpoints. Each checkpoint
     * consists of:
     * - A JSON file with metadata (simulation time, config, particle counts)
     * - A binary file with particle data (positions, velocities, etc.)
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel Kernel type (e.g., M4, C2)
     */
    template<class Tvec, template<class> class SPHKernel>
    class GSPHCheckpoint {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config = SolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;
        Config &solver_config;

        GSPHCheckpoint(ShamrockCtx &context, Config &solver_config)
            : context(context), solver_config(solver_config) {}

        /**
         * @brief Write a checkpoint to disk
         *
         * Saves the current simulation state to a checkpoint file.
         * Creates two files:
         * - {basename}.json: Metadata
         * - {basename}.bin: Binary particle data
         *
         * @param basename Base filename (without extension)
         * @param time Current simulation time
         * @param step Current timestep number
         */
        void write_checkpoint(const std::string &basename, Tscal time, u64 step);

        /**
         * @brief Read a checkpoint from disk
         *
         * Loads simulation state from a checkpoint file.
         * Reads both the JSON metadata and binary particle data.
         *
         * @param basename Base filename (without extension)
         * @param[out] time Simulation time from checkpoint
         * @param[out] step Timestep number from checkpoint
         */
        void read_checkpoint(const std::string &basename, Tscal &time, u64 &step);

        /**
         * @brief Check if a checkpoint file exists
         *
         * @param basename Base filename (without extension)
         * @return true if checkpoint exists
         */
        static bool checkpoint_exists(const std::string &basename);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        /**
         * @brief Write particle data to binary file
         */
        void write_binary_data(const std::string &filename);

        /**
         * @brief Read particle data from binary file
         */
        void read_binary_data(const std::string &filename, u64 expected_count);

        /**
         * @brief Generate JSON metadata for checkpoint
         */
        nlohmann::json generate_metadata(Tscal time, u64 step, u64 particle_count);

        /**
         * @brief Parse JSON metadata from checkpoint
         */
        void parse_metadata(const nlohmann::json &meta, Tscal &time, u64 &step, u64 &particle_count);
    };

} // namespace shammodels::gsph::modules
