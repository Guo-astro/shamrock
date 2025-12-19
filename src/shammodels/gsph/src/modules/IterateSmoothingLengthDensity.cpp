// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file IterateSmoothingLengthDensity.cpp
 * @author Guo (guo.yansong.ngy@gmail.com)
 * @brief Implements the GSPH IterateSmoothingLengthDensity module.
 *
 * This module performs ONE Newton-Raphson iteration for smoothing length,
 * designed to be used with LoopSmoothingLengthIter for proper outer-loop iteration.
 *
 * Key fix: Unlike the buggy inner-loop approach in the original compute_omega(),
 * this module is called from an outer loop that can rebuild the neighbor cache
 * when h changes significantly. This ensures density is computed correctly
 * at discontinuities like shock fronts and contact discontinuities.
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/modules/IterateSmoothingLengthDensity.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/patch/PatchDataField.hpp"

using namespace shammodels::gsph::modules;

template<class Tvec, class SPHKernel>
void IterateSmoothingLengthDensity<Tvec, SPHKernel>::_impl_evaluate_internal() {
    StackEntry stack_loc{};

    auto edges = get_edges();

    auto &thread_counts = edges.sizes.indexes;

    edges.neigh_cache.check_sizes(thread_counts);
    edges.positions.check_sizes(thread_counts);
    edges.old_h.check_sizes(thread_counts);
    edges.new_h.check_sizes(thread_counts);
    edges.eps_h.check_sizes(thread_counts);

    auto &neigh_cache = edges.neigh_cache.neigh_cache;
    auto &positions   = edges.positions.get_spans();
    auto &old_h       = edges.old_h.get_spans();
    auto &new_h       = edges.new_h.get_spans();
    auto &eps_h       = edges.eps_h.get_spans();

    // Get density and omega fields
    auto &density_field = edges.density;
    auto &omega_field   = edges.omega;

    // Ensure fields are allocated
    density_field.ensure_sizes(thread_counts);
    omega_field.ensure_sizes(thread_counts);

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    static constexpr Tscal Rkern = SPHKernel::Rkern;
    static constexpr u32 DIM     = 3; // 3D

    // Use distributed_data_kernel_call like the SPH version
    // This handles buffer access correctly
    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{neigh_cache, positions, old_h},
        sham::DDMultiRef{new_h, eps_h},
        thread_counts,
        [gpart_mass      = this->gpart_mass,
         h_evol_max      = this->h_evol_max,
         h_evol_iter_max = this->h_evol_iter_max](
            u32 id_a,
            auto ploop_ptrs,
            const Tvec *__restrict r,
            const Tscal *__restrict h_old,
            Tscal *__restrict h_new,
            Tscal *__restrict eps) {
            // Attach the neighbor looper on the cache
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            Tscal part_mass          = gpart_mass;
            Tscal h_max_tot_max_evol = h_evol_max;
            Tscal h_max_evol_p       = h_evol_iter_max;
            Tscal h_max_evol_m       = Tscal(1) / h_evol_iter_max;

            // Skip if already converged (eps < tolerance)
            if (eps[id_a] >= Tscal(0) && eps[id_a] < Tscal(1e-6)) {
                return;
            }

            Tvec xyz_a = r[id_a];
            Tscal h_a  = h_new[id_a]; // Use current iterate
            Tscal dint = h_a * h_a * Rkern * Rkern;

            // SPH density summation and dh derivative
            Tscal rho_sum = Tscal(0);
            Tscal sumdWdh = Tscal(0);

            particle_looper.for_each_object(id_a, [&](u32 id_b) {
                Tvec dr    = xyz_a - r[id_b];
                Tscal rab2 = sycl::dot(dr, dr);

                if (rab2 > dint) {
                    return; // Early return if too far
                }

                Tscal rab = sycl::sqrt(rab2);

                rho_sum += part_mass * SPHKernel::W_3d(rab, h_a);
                sumdWdh += part_mass * SPHKernel::dhW_3d(rab, h_a);
            });

            // Newton-Raphson step for h
            using namespace shamrock::sph;
            Tscal rho_ha      = rho_h(part_mass, h_a, SPHKernel::hfactd);
            Tscal new_h_val   = newtown_iterate_new_h(rho_ha, rho_sum, sumdWdh, h_a);

            // Clamp h change per iteration
            if (new_h_val < h_a * h_max_evol_m) {
                new_h_val = h_max_evol_m * h_a;
            }
            if (new_h_val > h_a * h_max_evol_p) {
                new_h_val = h_max_evol_p * h_a;
            }

            // Check total evolution limit
            Tscal ha_0 = h_old[id_a];

            if (new_h_val < ha_0 * h_max_tot_max_evol) {
                h_new[id_a] = new_h_val;
                eps[id_a]   = sycl::fabs(new_h_val - h_a) / ha_0;
            } else {
                // h grew too much - signal need for cache rebuild
                h_new[id_a] = ha_0 * h_max_tot_max_evol;
                eps[id_a]   = Tscal(-1); // Signal error condition
            }
        });

    // After h iteration, compute final density and omega for all particles
    // This uses the converged h values from the iteration above
    thread_counts.for_each([&](u64 patch_id, u32 cnt) {
        if (cnt == 0)
            return;

        auto &pcache     = neigh_cache.get(patch_id);
        auto &dens_field = density_field.get_field(patch_id);
        auto &omeg_field = omega_field.get_field(patch_id);

        // Get spans for positions and h
        auto &pos_span = positions.get(patch_id);
        auto &h_span   = new_h.get(patch_id);

        sham::DeviceQueue &q = dev_sched->get_queue();
        sham::EventList depends_list;

        auto ploop_ptrs  = pcache.get_read_access(depends_list);
        auto xyz_acc     = pos_span.get_read_access(depends_list);
        auto h_acc       = h_span.get_read_access(depends_list);
        auto density_acc = dens_field.get_buf().get_write_access(depends_list);
        auto omega_acc   = omeg_field.get_buf().get_write_access(depends_list);

        Tscal part_mass = this->gpart_mass;

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            shambase::parallel_for(
                cgh, cnt, "gsph_compute_density_omega", [=](u64 gid) {
                    u32 id_a = (u32) gid;

                    Tvec xyz_a = xyz_acc[id_a];
                    Tscal h_a  = h_acc[id_a];
                    Tscal dint = h_a * h_a * Rkern * Rkern;

                    // SPH density summation
                    Tscal rho_sum = Tscal(0);
                    Tscal sumdWdh = Tscal(0);

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        Tvec dr    = xyz_a - xyz_acc[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);

                        if (rab2 > dint) {
                            return;
                        }

                        Tscal rab = sycl::sqrt(rab2);

                        rho_sum += part_mass * SPHKernel::W_3d(rab, h_a);
                        sumdWdh += part_mass * SPHKernel::dhW_3d(rab, h_a);
                    });

                    // Store density
                    density_acc[id_a] = sycl::max(rho_sum, Tscal(1e-30));

                    // Compute omega (grad-h correction factor)
                    // omega = 1 / (1 + h/(D*rho) * dh_rho)
                    Tscal omega_val = Tscal(1);
                    if (rho_sum > Tscal(1e-30)) {
                        omega_val
                            = Tscal(1) / (Tscal(1) + h_a / (Tscal(DIM) * rho_sum) * sumdWdh);
                        omega_val = sycl::clamp(omega_val, Tscal(0.5), Tscal(1.5));
                    }
                    omega_acc[id_a] = omega_val;
                });
        });

        // Complete event states for all accessed buffers
        pcache.complete_event_state({e});
        pos_span.complete_event_state(e);
        h_span.complete_event_state(e);
        dens_field.get_buf().complete_event_state(e);
        omeg_field.get_buf().complete_event_state(e);
    });
}

template<class Tvec, class SPHKernel>
std::string IterateSmoothingLengthDensity<Tvec, SPHKernel>::_impl_get_tex() const {
    auto sizes       = get_ro_edge_base(0).get_tex_symbol();
    auto neigh_cache = get_ro_edge_base(1).get_tex_symbol();
    auto positions   = get_ro_edge_base(2).get_tex_symbol();
    auto old_h       = get_ro_edge_base(3).get_tex_symbol();
    auto new_h       = get_rw_edge_base(0).get_tex_symbol();
    auto eps_h       = get_rw_edge_base(1).get_tex_symbol();
    auto density     = get_rw_edge_base(2).get_tex_symbol();
    auto omega       = get_rw_edge_base(3).get_tex_symbol();

    std::string tex = R"tex(
            GSPH Iterate smoothing length, density, and omega

            \begin{align}
            \rho_i &= \sum_{j \in \mathcal{N}_i} m_j W(r_{ij}, h_i) \\
            \frac{\partial \rho_i}{\partial h_i} &= \sum_{j \in \mathcal{N}_i} m_j \frac{\partial W}{\partial h}(r_{ij}, h_i) \\
            h_i^{\rm new} &= h_i - \frac{\rho_i - \rho_h(m_i, h_i)}{\frac{\partial \rho_i}{\partial h_i} + \frac{3\rho_h(m_i, h_i)}{h_i}} \\
            \Omega_i &= \frac{1}{1 + \frac{h_i}{3\rho_i} \frac{\partial \rho_i}{\partial h_i}} \\
            \epsilon_i &= \frac{|h_i^{\rm new} - h_i|}{h_i^{\rm old}}
            \end{align}

            where:
            \begin{itemize}
            \item $\mathcal{N}_i$ is the set of neighbors of particle $i$
            \item $W(r, h)$ is the SPH kernel function
            \item $\rho_h(m, h) = m \left(\frac{h_{\rm fact}}{h}\right)^3$ is the target density
            \item $\Omega$ is the grad-h correction factor for GSPH
            \end{itemize}

            Input: ${sizes}$, ${neigh_cache}$, ${positions}$, ${old_h}$
            Output: ${new_h}$, ${eps_h}$, ${density}$, ${omega}$
        )tex";

    shambase::replace_all(tex, "{sizes}", sizes);
    shambase::replace_all(tex, "{neigh_cache}", neigh_cache);
    shambase::replace_all(tex, "{positions}", positions);
    shambase::replace_all(tex, "{old_h}", old_h);
    shambase::replace_all(tex, "{new_h}", new_h);
    shambase::replace_all(tex, "{eps_h}", eps_h);
    shambase::replace_all(tex, "{density}", density);
    shambase::replace_all(tex, "{omega}", omega);

    return tex;
}

// Template instantiations for all supported kernels
template class shammodels::gsph::modules::IterateSmoothingLengthDensity<f64_3, shammath::M4<f64>>;
template class shammodels::gsph::modules::IterateSmoothingLengthDensity<f64_3, shammath::M6<f64>>;
template class shammodels::gsph::modules::IterateSmoothingLengthDensity<f64_3, shammath::M8<f64>>;

template class shammodels::gsph::modules::IterateSmoothingLengthDensity<f64_3, shammath::C2<f64>>;
template class shammodels::gsph::modules::IterateSmoothingLengthDensity<f64_3, shammath::C4<f64>>;
template class shammodels::gsph::modules::IterateSmoothingLengthDensity<f64_3, shammath::C6<f64>>;
