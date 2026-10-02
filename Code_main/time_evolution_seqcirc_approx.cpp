/*
 * time_evolution_seqcirc_approx.cpp
 *
 * Approximated TWO-simultaneously-circulating-variant renewal equation solver.
 *
 * Scenario: variants emerge sequentially but with overlap, so at any time
 * EXACTLY 2 variants are co-circulating (except the very first period).
 *
 * Active periods:
 *   Period 1:  0 < t <= T^em_2        only variant 1 active
 *   Period c:  T^em_c < t <= T^em_{c+1}   variants c-1 and c active  (c >= 2)
 *   Final:     T^em_{N} < t <= t_max      variants N-1 and N active
 *
 * Each variant c is active during  T^em_c < t <= T^em_{c+2},
 * ensuring exactly 2 co-circulating variants in all periods except the first.
 *
 * -------------------------------------------------------------------
 * KEY EQUATIONS  (from the PDF, second-page "corrected" formulation)
 * -------------------------------------------------------------------
 *
 * Force of infection  (same renewal form as before, one per active variant):
 *
 *   Lambda_nu(t) = R^nu_empty(t) * SUM_{h: nu not in h} [
 *                    sigma_tilde^nu_h * sigma^nu_h
 *                    * INT_0^inf g^nu_h(tau) * s_h(t-tau) * Lambda_nu(t-tau) dtau
 *                  ] + R^nu_empty(t) * F_nu(t)
 *
 * Infection incidence:
 *   I^nu_h(t) = s_h(t) * sigma^nu_h * Lambda_nu(t)
 *
 * Recovery flux:
 *   H^u_h(t) = INT_0^t I^u_h(t-tau) * gamma^u_h(tau) dtau
 *            = sigma^u_h * INT_0^t s_h(t-tau) * Lambda_u(t-tau) * gamma^u_h(tau) dtau
 *            + seed_gamma term
 *
 * Susceptibility ODEs for period  T^em_c < t <= T^em_{c+1}
 * (two circulating variants: c-1 and c):
 *
 *   Case 1:  c ∈ h  &  c-1 ∈ h
 *     ṡ_h = + H^c_{h-{c}}(t) + H^{c-1}_{h-{c-1}}(t)
 *
 *   Case 2:  c ∈ h  &  c-1 ∉ h
 *     ṡ_h = - s_h(t) * sigma^{c-1}_h * Lambda_{c-1}(t)  +  H^c_{h-{c}}(t)
 *
 *   Case 3:  c ∉ h  &  c-1 ∈ h
 *     ṡ_h = - s_h(t) * sigma^c_h * Lambda_c(t)  +  H^{c-1}_{h-{c-1}}(t)
 *
 *   Case 4:  c ∉ h  &  c-1 ∉ h
 *     ṡ_h = - s_h(t) * sigma^{c-1}_h * Lambda_{c-1}(t)
 *            - s_h(t) * sigma^c_h     * Lambda_c(t)
 *
 * Special first period  0 < t <= T^em_2  (only variant 1 circulating):
 *   handled exactly as in time_evolution_approx.cpp (single-circulating case).
 *
 * -------------------------------------------------------------------
 * DISCRETISATION  (Euler, consistent with working approx code)
 * -------------------------------------------------------------------
 *
 *  Lambda_nu[i] is computed from s[k][m] and Lambda[nu][m] for m < i
 *  (convolution runs from emergence index to i-1 to avoid circular dependency).
 *
 *  s_h[i] is updated from s_h[i-1] plus dt * (rhs evaluated at i-1).
 *
 *  H^u_h(t_i) needed for ṡ (case 1,2,3):
 *    H^u_{h'}(t_i) = sigma^u_{h'} * dt * SUM_{m=i_em_u}^{i} gamma^u_{h'}(tau_{i-m}) * s_{h'}(m) * Lambda_u(m)
 *                  + imp_gamma(u, h, i)          <- seed import term
 *    where h' = h - {u}.
 *
 * -------------------------------------------------------------------
 * Compile (pybind11 module):
 *   g++ -O3 -std=c++17 -shared -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       time_evolution_seqcirc_approx.cpp \
 *       -o infectionRenewalEq_seqcirc_approx$(python3-config --extension-suffix)
 * -------------------------------------------------------------------
 * Compile (pybind11 module):
 * % arch -x86_64 c++ -O3 -march=native -ffast-math -funroll-loops -Wall -shared -std=c++17 -fPIC \
 *    `python3 -m pybind11 --includes` time_evolution_seqcirc_approx.cpp \ 
 *    -o infectionRenewalEq_seqcirc_approx$(python3-config --extension-suffix) \
 *   -undefined dynamic_lookup
 */

#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

// ---------------------------------------------------------------------------
// Helper: build map  sorted-vector<int> -> flat index
// ---------------------------------------------------------------------------
static map<vector<int>, int> build_h_index_map(const vector<set<int>>& h_values_all) {
    map<vector<int>, int> m;
    for (int k = 0; k < (int)h_values_all.size(); ++k) {
        vector<int> key(h_values_all[k].begin(), h_values_all[k].end());
        m[key] = k;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------
py::tuple time_evolution_seqcirc_approx(
    py::array_t<double> t_values_arr,
    py::array_t<double> J_values_arr,            // [num_nu, n_steps]  seed for Lambda
    py::array_t<double> s_values_arr,            // [num_h,  n_steps]  modified in-place
    py::array_t<double> R_nu_arr,                // [num_nu, n_steps]
    py::array_t<double> sigma_nu_h_arr,          // [num_nu, num_h]
    py::array_t<double> tilde_sigma_nu_h_arr,    // [num_nu, num_h]
    py::array_t<double> g_table_arr,             // [num_nu, num_h, n_steps]
    py::array_t<double> gamma_table_arr,         // [num_nu, num_h, n_steps]
    py::array_t<double> IntgImpG_Emerg_arr,      // [num_nu, n_steps]
    py::array_t<double> IntgImpGamma_Emerg_arr,  // [num_nu, num_h, n_steps]
    vector<int>       nu_values,
    vector<set<int>>  h_values_all,
    map<vector<int>, int> h_index_map_py,        // kept for API parity
    vector<double>    t_emerge,                   // T^em_c for each variant
    int               n_overlap = 2              // number of co-circulating variants (1 or 2)
) {
    // ---- array accessors ----
    auto t_values    = t_values_arr.unchecked<1>();
    auto s_arr       = s_values_arr.mutable_unchecked<2>();   // [h, t]
    auto R_nu        = R_nu_arr.unchecked<2>();                // [nu, t]
    auto sigma       = sigma_nu_h_arr.unchecked<2>();          // [nu, h]
    auto tilde_sigma = tilde_sigma_nu_h_arr.unchecked<2>();    // [nu, h]
    auto g_tab       = g_table_arr.unchecked<3>();             // [nu, h, t]
    auto gam_tab     = gamma_table_arr.unchecked<3>();         // [nu, h, t]
    auto imp_g       = IntgImpG_Emerg_arr.unchecked<2>();      // [nu, t]
    auto imp_gamma   = IntgImpGamma_Emerg_arr.mutable_unchecked<3>(); // [nu, h, t]

    const int n_steps = (int)t_values.shape(0);
    const int num_nu  = (int)nu_values.size();
    const int num_h   = (int)h_values_all.size();
    const double dt   = (n_steps > 1) ? (t_values(1) - t_values(0)) : 1.0;

    // ---- h_index helper ----
    auto h_idx = build_h_index_map(h_values_all);
    auto get_h_idx = [&](const set<int>& h) -> int {
        vector<int> key(h.begin(), h.end());
        auto it = h_idx.find(key);
        if (it == h_idx.end()) throw runtime_error("h not found in h_index_map");
        return it->second;
    };

    // ---- nu_index helper (variant label -> position in nu_values) ----
    map<int, int> nu_pos;
    for (int j = 0; j < num_nu; ++j) nu_pos[nu_values[j]] = j;

    // ---- emergence time indices ----
    vector<int> i_em(num_nu, 0);
    for (int j = 0; j < num_nu; ++j)
        for (int i = 0; i < n_steps; ++i)
            if (t_values(i) >= t_emerge[j] - 1e-12) { i_em[j] = i; break; }

    // ---- Lambda [num_nu x n_steps] ----
    vector<vector<double>> Lambda(num_nu, vector<double>(n_steps, 0.0));

    // ---- precompute H_h^u convolution integrals ----
    // H_conv[j][k][i] = sigma^u_{h'} * dt * SUM_{m=i_em[j]}^{i}
    //                       gamma^u_{h'}(tau_{i-m}) * s_{h'}(m) * Lambda_u(m)
    // where h' = h_values_all[k] \ {nu_values[j]}
    // This is built incrementally inside the time loop.
    // We store H_conv as [num_nu][num_h] running sums, updated each step.
    // (We need the full convolution up to current i, so we accumulate.)

    // auto start = chrono::high_resolution_clock::now();

    // ====================================================================
    // MAIN TIME LOOP  (outer loop over time steps)
    // ====================================================================
    for (int i = 1; i < n_steps; ++i) {

        // ----------------------------------------------------------------
        // Determine which variants are active at step i.
        // Variant j is active during  T^em_j < t <= T^em_{j+n_overlap}
        // 
        // n_overlap = 1: single-circulating (T^em_j < t <= T^em_{j+1})
        //   Each variant active only until next one emerges
        // n_overlap = 2: two-circulating (T^em_j < t <= T^em_{j+2})
        //   Variants c-1 and c co-circulate in period T^em_c < t <= T^em_{c+1}
        // ----------------------------------------------------------------
        vector<int> active;   // indices into nu_values of currently active variants
        for (int j = 0; j < num_nu; ++j) {
            bool started = (i > i_em[j]);
            bool not_ended = (j + n_overlap >= num_nu) || (i <= i_em[j + n_overlap]);
            if (started && not_ended) active.push_back(j);
        }

        // ----------------------------------------------------------------
        // Step 1: compute Lambda_nu(t_i) for all active variants
        //         using s_h values already updated up to step i-1,
        //         and Lambda values from steps < i.
        // ----------------------------------------------------------------
        for (int jj = 0; jj < (int)active.size(); ++jj) {
            int j  = active[jj];
            int nu = nu_values[j];
            int i0 = i_em[j];   // start of convolution for variant j

            double R_empty_i = R_nu(j, i);
            double sum_conv  = 0.0;

            for (int k = 0; k < num_h; ++k) {
                if (h_values_all[k].count(nu) != 0) continue;  // skip h with nu in h
                double sig  = sigma(j, k);
                double tsig = tilde_sigma(j, k);
                if (sig == 0.0 || tsig == 0.0) continue;

                double conv = 0.0;
                for (int m = i0; m <= i-1; ++m) {   // m < i: use only past values
                    int tau_idx = i - m;
                    conv += g_tab(j, k, tau_idx) * s_arr(k, m) * Lambda[j][m];
                }
                conv *= dt;
                sum_conv += tsig * sig * conv;
            }

            Lambda[j][i] = R_empty_i * sum_conv + R_empty_i * imp_g(j, i);
        }

        // ----------------------------------------------------------------
        // Step 2: update s_h(t_i) for all histories h
        //
        // The update is:
        //   s_h[i] = s_h[i-1] + dt * ṡ_h[i-1]
        //
        // where ṡ_h is determined by which active variants are in/not-in h.
        //
        // For a general active set {nu_1, nu_2, ...} the rhs is:
        //   ṡ_h = SUM_{u in active, u in h}     H^u_{h-{u}}(t)
        //       + SUM_{u in active, u not in h} (- s_h(t) * sigma^u_h * Lambda_u(t))
        
        // H_full[j][k] = sigma^u_{h} * dt * SUM_{m=i0}^{i}
        //                   gamma^u_{h}(tau_{i-m}) * s_{h}(m) * Lambda_u(m)
        //                + imp_gamma(j, k, i)
        //
        // This generalises the 4 cases in the PDF to arbitrary active sets.
        // (For |active|=1 it reduces to the single-circulating code;
        //  for |active|=2 it gives the 4 PDF cases exactly.)
        // ----------------------------------------------------------------
        for (int k = 0; k < num_h; ++k) {
            double ds = 0.0;

            for (int jj = 0; jj < (int)active.size(); ++jj) {
                int j  = active[jj];
                int nu = nu_values[j];

                if (h_values_all[k].count(nu) != 0) {
                    // nu in h  →  +H^nu_{h-{nu}}(t_{i-1})
                    // For simplicity we recompute H at i-1 here.
                    set<int> h_minus_c = h_values_all[k];
                    h_minus_c.erase(nu);
                    int k_minus_c    = get_h_idx(h_minus_c);
                    double sig_minus = sigma(j, k_minus_c);
                    int i0 = i_em[j];

                    double conv_prev = 0.0;
                    for (int m = i0; m <= i - 1; ++m) {
                        int tau_idx = i - m;
                        conv_prev += gam_tab(j, k_minus_c, tau_idx) * s_arr(k_minus_c, m) * Lambda[j][m];
                    }
                    conv_prev *= dt;
                    double H_val = sig_minus * conv_prev + imp_gamma(j, k_minus_c, i-1);
                    ds += H_val;

                } else {
                    // nu NOT in h  →  - s_h(i-1) * sigma^nu_h * Lambda_nu(i-1)
                    double sig = sigma(j, k);
                    ds -= s_arr(k, i - 1) * sig * Lambda[j][i - 1];
                }
            }           

            s_arr(k, i) = s_arr(k, i - 1) + dt * ds;
        }
    }

    // auto end = chrono::high_resolution_clock::now();
    // chrono::duration<double> elapsed = end - start;
    // cout << "Approx2 execution time: " << elapsed.count() << " seconds" << endl;

    // ---- pack Lambda -> numpy ----
    py::array_t<double> Lambda_out({num_nu, n_steps});
    auto lam_buf = Lambda_out.mutable_unchecked<2>();
    for (int j = 0; j < num_nu; ++j)
        for (int i = 0; i < n_steps; ++i)
            lam_buf(j, i) = Lambda[j][i];

    return py::make_tuple(Lambda_out, s_values_arr);
}

// ---------------------------------------------------------------------------
// Pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(infectionRenewalEq_seqcirc_approx, m) {
    m.doc() = "Approximated two-simultaneously-circulating-variant renewal equation solver";

    m.def("time_evolution_seqcirc_approx",
          &time_evolution_seqcirc_approx,
          py::arg("t_values"),
          py::arg("J_values"),
          py::arg("s_values"),
          py::arg("R_nu"),
          py::arg("sigma_nu_h"),
          py::arg("tilde_sigma_nu_h"),
          py::arg("g_table"),
          py::arg("gamma_table"),
          py::arg("IntgImpG_Emerg"),
          py::arg("IntgImpGamma_Emerg"),
          py::arg("nu_values"),
          py::arg("h_values_all"),
          py::arg("h_index_map"),
          py::arg("t_emerge"),
          py::arg("n_overlap") = 2,  // default: 2 co-circulating variants
          R"pbdoc(
Two-simultaneously-circulating-variant approximation solver.

Supports both single-circulating (n_overlap=1) and two-circulating (n_overlap=2)
scenarios via the n_overlap parameter.

n_overlap = 1 (single-circulating):
  Variant j active during  T^em_j < t <= T^em_{j+1}
  Only 1 variant circulates at a time (like time_evolution_approx.cpp)

n_overlap = 2 (two-circulating, default):
  Variant j active during  T^em_j < t <= T^em_{j+2}
  For period  T^em_c < t <= T^em_{c+1}  variants c-1 and c are both active


For the period  T^em_c < t <= T^em_{c+1}  variants c-1 and c are both
active.  The susceptibility ODEs are (PDF page 2, corrected formulation):

  Case 1  (c ∈ h  &  c-1 ∈ h):
      ṡ_h = + H^c_{h-{c}}(t)  +  H^{c-1}_{h-{c-1}}(t)

  Case 2  (c ∈ h  &  c-1 ∉ h):
      ṡ_h = - s_h(t) * sigma^{c-1}_h * Lambda_{c-1}(t)  +  H^c_{h-{c}}(t)

  Case 3  (c ∉ h  &  c-1 ∈ h):
      ṡ_h = - s_h(t) * sigma^c_h * Lambda_c(t)  +  H^{c-1}_{h-{c-1}}(t)

  Case 4  (c ∉ h  &  c-1 ∉ h):
      ṡ_h = - s_h(t) * sigma^{c-1}_h * Lambda_{c-1}(t)
             - s_h(t) * sigma^c_h     * Lambda_c(t)

With:
  H^u_h(t)     = sigma^u_{h'} * INT_0^t gamma^u_{h'}(tau) * s_{h'}(t-tau) * Lambda_u(t-tau) dtau
                 + seed_gamma_import(u, h, t)
  Lambda_nu(t) = R^nu_empty(t) * SUM_{h: nu not in h} [
                   sigma_tilde^nu_h * sigma^nu_h
                   * INT_0^t g^nu_h(tau) * s_h(t-tau) * Lambda_nu(t-tau) dtau ]
               + R^nu_empty(t) * F_nu(t)

The outer loop is over time; all active-variant quantities are updated
at each time step, making it straightforward to extend to more than 2
simultaneously circulating variants.

Parameters  (identical to time_evolution_approx)
----------
t_values, J_values, s_values, R_nu, sigma_nu_h, tilde_sigma_nu_h,
g_table, gamma_table, IntgImpG_Emerg, IntgImpGamma_Emerg,
nu_values, h_values_all, h_index_map, t_emerge

Returns
-------
(Lambda_values [num_nu, n_steps], s_values [num_h, n_steps])
          )pbdoc");
}