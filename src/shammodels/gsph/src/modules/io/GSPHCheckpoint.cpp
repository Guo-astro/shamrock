// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GSPHCheckpoint.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Implementation of GSPH checkpoint/restart functionality
 */

#include "shambase/exception.hpp"
#include "shammodels/gsph/modules/io/GSPHCheckpoint.hpp"
#include "shammath/sphkernels.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamsys/legacy/log.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    bool GSPHCheckpoint<Tvec, SPHKernel>::checkpoint_exists(const std::string &basename) {
        std::string json_file = basename + ".json";
        std::string bin_file  = basename + ".bin";
        return std::filesystem::exists(json_file) && std::filesystem::exists(bin_file);
    }

    template<class Tvec, template<class> class SPHKernel>
    nlohmann::json GSPHCheckpoint<Tvec, SPHKernel>::generate_metadata(
        Tscal time, u64 step, u64 particle_count) {

        nlohmann::json meta;

        // Simulation state
        meta["time"]           = time;
        meta["step"]           = step;
        meta["particle_count"] = particle_count;

        // Solver configuration
        meta["config"]["gamma"]                = solver_config.gamma;
        meta["config"]["gpart_mass"]           = solver_config.gpart_mass;
        meta["config"]["cfl_cour"]             = solver_config.cfl_config.cfl_cour;
        meta["config"]["cfl_force"]            = solver_config.cfl_config.cfl_force;
        meta["config"]["htol_up_coarse_cycle"] = solver_config.htol_up_coarse_cycle;
        meta["config"]["htol_up_fine_cycle"]   = solver_config.htol_up_fine_cycle;

        // EOS type - use variant get_if
        using EOSAdiabatic  = typename Config::EOSConfig::Adiabatic;
        using EOSIsothermal = typename Config::EOSConfig::Isothermal;
        if (std::get_if<EOSAdiabatic>(&solver_config.eos_config.config)) {
            meta["config"]["eos_type"] = "adiabatic";
        } else if (std::get_if<EOSIsothermal>(&solver_config.eos_config.config)) {
            meta["config"]["eos_type"] = "isothermal";
        }

        // Riemann solver type - use variant get_if
        using RiemannIterative = typename Config::RiemannConfig::Iterative;
        using RiemannHLLC      = typename Config::RiemannConfig::HLLC;
        if (std::get_if<RiemannIterative>(&solver_config.riemann_config.config)) {
            meta["config"]["riemann_type"] = "iterative";
        } else if (std::get_if<RiemannHLLC>(&solver_config.riemann_config.config)) {
            meta["config"]["riemann_type"] = "hllc";
        }

        // Field flags
        meta["config"]["has_uint"] = solver_config.has_field_uint();

        // Version info for compatibility
        meta["version"]   = 1;
        meta["format"]    = "gsph_checkpoint";
        meta["dimension"] = dim;

        return meta;
    }

    template<class Tvec, template<class> class SPHKernel>
    void GSPHCheckpoint<Tvec, SPHKernel>::parse_metadata(
        const nlohmann::json &meta, Tscal &time, u64 &step, u64 &particle_count) {

        // Check version compatibility
        int version = meta.at("version").get<int>();
        if (version != 1) {
            shambase::throw_with_loc<std::runtime_error>(
                "Unsupported checkpoint version: " + std::to_string(version));
        }

        // Check dimension compatibility
        int file_dim = meta.at("dimension").get<int>();
        if (file_dim != dim) {
            shambase::throw_with_loc<std::runtime_error>(
                "Dimension mismatch: checkpoint has " + std::to_string(file_dim) + ", expected "
                + std::to_string(dim));
        }

        // Read simulation state
        time           = meta.at("time").get<Tscal>();
        step           = meta.at("step").get<u64>();
        particle_count = meta.at("particle_count").get<u64>();

        // Verify config compatibility
        Tscal file_gamma = meta.at("config").at("gamma").get<Tscal>();
        if (std::abs(file_gamma - solver_config.gamma) > Tscal(1e-10)) {
            logger::warn_ln(
                "GSPHCheckpoint",
                "gamma mismatch: file has",
                file_gamma,
                ", config has",
                solver_config.gamma);
        }
    }

    template<class Tvec, template<class> class SPHKernel>
    void GSPHCheckpoint<Tvec, SPHKernel>::write_binary_data(const std::string &filename) {

        using namespace shamrock::patch;

        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            shambase::throw_with_loc<std::runtime_error>(
                "Cannot open file for writing: " + filename);
        }

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
        const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");
        const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
        const bool has_uint       = solver_config.has_field_uint();
        const u32 iuint           = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;
        const u32 iduint          = has_uint ? pdl.get_field_idx<Tscal>("duint") : 0;

        // Collect all particle data from patches
        std::vector<Tvec> all_xyz;
        std::vector<Tvec> all_vxyz;
        std::vector<Tvec> all_axyz;
        std::vector<Tscal> all_hpart;
        std::vector<Tscal> all_uint;
        std::vector<Tscal> all_duint;

        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                u64 cnt = pdat.get_obj_cnt();

                // Read data from device
                auto xyz_span   = pdat.get_field_buf_ref<Tvec>(ixyz).copy_to_stdvec();
                auto vxyz_span  = pdat.get_field_buf_ref<Tvec>(ivxyz).copy_to_stdvec();
                auto axyz_span  = pdat.get_field_buf_ref<Tvec>(iaxyz).copy_to_stdvec();
                auto hpart_span = pdat.get_field_buf_ref<Tscal>(ihpart).copy_to_stdvec();

                for (u64 i = 0; i < cnt; i++) {
                    all_xyz.push_back(xyz_span[i]);
                    all_vxyz.push_back(vxyz_span[i]);
                    all_axyz.push_back(axyz_span[i]);
                    all_hpart.push_back(hpart_span[i]);
                }

                if (has_uint) {
                    auto uint_span  = pdat.get_field_buf_ref<Tscal>(iuint).copy_to_stdvec();
                    auto duint_span = pdat.get_field_buf_ref<Tscal>(iduint).copy_to_stdvec();
                    for (u64 i = 0; i < cnt; i++) {
                        all_uint.push_back(uint_span[i]);
                        all_duint.push_back(duint_span[i]);
                    }
                }
            });

        // Write particle count
        u64 count = all_xyz.size();
        file.write(reinterpret_cast<const char *>(&count), sizeof(count));

        // Write arrays
        file.write(reinterpret_cast<const char *>(all_xyz.data()), count * sizeof(Tvec));
        file.write(reinterpret_cast<const char *>(all_vxyz.data()), count * sizeof(Tvec));
        file.write(reinterpret_cast<const char *>(all_axyz.data()), count * sizeof(Tvec));
        file.write(reinterpret_cast<const char *>(all_hpart.data()), count * sizeof(Tscal));

        if (has_uint) {
            file.write(reinterpret_cast<const char *>(all_uint.data()), count * sizeof(Tscal));
            file.write(reinterpret_cast<const char *>(all_duint.data()), count * sizeof(Tscal));
        }

        file.close();
        logger::info_ln("GSPHCheckpoint", "Wrote", count, "particles to", filename);
    }

    template<class Tvec, template<class> class SPHKernel>
    void GSPHCheckpoint<Tvec, SPHKernel>::read_binary_data(
        const std::string &filename, u64 expected_count) {

        using namespace shamrock::patch;

        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            shambase::throw_with_loc<std::runtime_error>(
                "Cannot open file for reading: " + filename);
        }

        // Read particle count
        u64 count;
        file.read(reinterpret_cast<char *>(&count), sizeof(count));

        if (count != expected_count) {
            shambase::throw_with_loc<std::runtime_error>(
                "Particle count mismatch: file has " + std::to_string(count) + ", expected "
                + std::to_string(expected_count));
        }

        // Read arrays
        std::vector<Tvec> all_xyz(count);
        std::vector<Tvec> all_vxyz(count);
        std::vector<Tvec> all_axyz(count);
        std::vector<Tscal> all_hpart(count);

        file.read(reinterpret_cast<char *>(all_xyz.data()), count * sizeof(Tvec));
        file.read(reinterpret_cast<char *>(all_vxyz.data()), count * sizeof(Tvec));
        file.read(reinterpret_cast<char *>(all_axyz.data()), count * sizeof(Tvec));
        file.read(reinterpret_cast<char *>(all_hpart.data()), count * sizeof(Tscal));

        const bool has_uint = solver_config.has_field_uint();
        std::vector<Tscal> all_uint(count);
        std::vector<Tscal> all_duint(count);

        if (has_uint) {
            file.read(reinterpret_cast<char *>(all_uint.data()), count * sizeof(Tscal));
            file.read(reinterpret_cast<char *>(all_duint.data()), count * sizeof(Tscal));
        }

        file.close();

        // Clear existing particles and add loaded ones
        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
        const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");
        const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
        const u32 iuint           = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;
        const u32 iduint          = has_uint ? pdl.get_field_idx<Tscal>("duint") : 0;

        // For simplicity, load all particles into a single patch
        // In production, you'd want to distribute across patches based on position
        scheduler().for_each_patch_data([&](u64 id_patch, Patch &p, PatchDataLayer &pdat) {
            pdat.resize(0); // Clear existing
        });

        // Get first patch and load data
        bool loaded = false;
        scheduler().for_each_patch_data([&](u64 id_patch, Patch &p, PatchDataLayer &pdat) {
            if (loaded)
                return;

            pdat.resize(count);

            pdat.get_field_buf_ref<Tvec>(ixyz).copy_from_stdvec(all_xyz);
            pdat.get_field_buf_ref<Tvec>(ivxyz).copy_from_stdvec(all_vxyz);
            pdat.get_field_buf_ref<Tvec>(iaxyz).copy_from_stdvec(all_axyz);
            pdat.get_field_buf_ref<Tscal>(ihpart).copy_from_stdvec(all_hpart);

            if (has_uint) {
                pdat.get_field_buf_ref<Tscal>(iuint).copy_from_stdvec(all_uint);
                pdat.get_field_buf_ref<Tscal>(iduint).copy_from_stdvec(all_duint);
            }

            loaded = true;
        });

        logger::info_ln("GSPHCheckpoint", "Loaded", count, "particles from", filename);
    }

    template<class Tvec, template<class> class SPHKernel>
    void GSPHCheckpoint<Tvec, SPHKernel>::write_checkpoint(
        const std::string &basename, Tscal time, u64 step) {

        // Count total particles
        u64 total_count = 0;
        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                total_count += pdat.get_obj_cnt();
            });

        // Generate and write metadata
        nlohmann::json meta = generate_metadata(time, step, total_count);

        std::string json_file = basename + ".json";
        std::ofstream json_out(json_file);
        if (!json_out) {
            shambase::throw_with_loc<std::runtime_error>(
                "Cannot open file for writing: " + json_file);
        }
        json_out << std::setw(2) << meta << std::endl;
        json_out.close();

        // Write binary particle data
        std::string bin_file = basename + ".bin";
        write_binary_data(bin_file);

        logger::info_ln(
            "GSPHCheckpoint",
            "Checkpoint written: time =",
            time,
            ", step =",
            step,
            ", particles =",
            total_count);
    }

    template<class Tvec, template<class> class SPHKernel>
    void GSPHCheckpoint<Tvec, SPHKernel>::read_checkpoint(
        const std::string &basename, Tscal &time, u64 &step) {

        // Read metadata
        std::string json_file = basename + ".json";
        std::ifstream json_in(json_file);
        if (!json_in) {
            shambase::throw_with_loc<std::runtime_error>(
                "Cannot open file for reading: " + json_file);
        }

        nlohmann::json meta;
        json_in >> meta;
        json_in.close();

        u64 particle_count;
        parse_metadata(meta, time, step, particle_count);

        // Read binary particle data
        std::string bin_file = basename + ".bin";
        read_binary_data(bin_file, particle_count);

        logger::info_ln(
            "GSPHCheckpoint",
            "Checkpoint loaded: time =",
            time,
            ", step =",
            step,
            ", particles =",
            particle_count);
    }

} // namespace shammodels::gsph::modules

// Explicit template instantiations
using namespace shammath;
template class shammodels::gsph::modules::GSPHCheckpoint<f64_3, M4>;
template class shammodels::gsph::modules::GSPHCheckpoint<f64_3, M6>;
template class shammodels::gsph::modules::GSPHCheckpoint<f64_3, M8>;
template class shammodels::gsph::modules::GSPHCheckpoint<f64_3, C2>;
template class shammodels::gsph::modules::GSPHCheckpoint<f64_3, C4>;
template class shammodels::gsph::modules::GSPHCheckpoint<f64_3, C6>;
