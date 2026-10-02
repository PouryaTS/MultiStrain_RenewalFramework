
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <chrono>
#include <iostream>

namespace py = pybind11;
using namespace std;

//First, define the conversion function BEFORE using it
std::map<std::set<int>, int> convert_h_index_map(const py::dict& input) {
    std::map<std::set<int>, int> output;
    for (auto item : input) {
        std::set<int> key;
        for (auto element : item.first.cast<py::tuple>()) {
            key.insert(element.cast<int>());
        }
        output[key] = item.second.cast<int>();
    }
    return output;
}


void time_evolution_coupled(
    py::array_t<double> t_values, 
    py::array_t<double> J_values, 
    py::array_t<double> s_values,
    py::array_t<double> R_nu,
    py::array_t<double> sigma_nu_h,
    py::array_t<double> tilde_sigma_nu_h,
    py::array_t<double> g_table, 
    py::array_t<double> gamma_table, 
    py::array_t<double> IntgImpG,
    py::array_t<double> IntgImpGamma,
    vector<int> nu_values,
    vector<set<int>> h_values_all,  
    const py::dict& h_index_map_py  // Accept dict[tuple, int]
) {
    auto start = std::chrono::high_resolution_clock::now();
    auto t_vals = t_values.unchecked<1>();
    auto J_vals = J_values.mutable_unchecked<2>();
    auto s_vals = s_values.mutable_unchecked<2>();
    auto R_vals = R_nu.unchecked<2>();
    auto sigma_vals = sigma_nu_h.unchecked<2>();
    auto tilde_sigma_vals = tilde_sigma_nu_h.unchecked<2>();
    auto g_table_vals = g_table.mutable_unchecked<3>();
    auto gamma_table_vals = gamma_table.mutable_unchecked<3>();
    auto IntgImpG_vals = IntgImpG.mutable_unchecked<2>();
    auto IntgImpGamma_vals = IntgImpGamma.mutable_unchecked<3>();

    std::map<std::set<int>, int> h_index_map = convert_h_index_map(h_index_map_py);

    int n_steps = t_vals.shape(0);
    int num_nu = J_vals.shape(0);
    int num_h = h_values_all.size();
    double dt = t_vals(1) - t_vals(0);

    //std::vector<std::vector<double>> H_vals(num_nu, std::vector<double>(num_h));
    //py::array_t<double> H_val = py::array::zeros({num_nu, num_h, n_steps}, py::dtype::of<double>());
    //auto H_val = py::array_t<double>::zeros({num_nu, num_h, n_steps});
    //py::array_t<double> H_values = py::array_t<double>::zeros({num_nu, num_h, n_steps});
    py::array_t<double> H_values({num_nu, num_h, n_steps});
    auto buf = H_values.request();
    std::memset(buf.ptr, 0, buf.size * sizeof(double));
    auto H_vals = H_values.mutable_unchecked<3>();

    for (int i = 1; i < n_steps; ++i) {
        double t = t_vals(i);
        // ====================================================================
        // STEP 1: Compute J_vals[j, i] (Force of Infection / Lambda)
        // ====================================================================
        // Equation: J[j,i] = R[j,i] * SUM_k tilde_sigma[j,k] * integral_k
        //           where integral_k = SUM_{m=0}^{i-1} g[tau_{i-m}] * s[m] * sigma[j,k] * J[m]
        // ====================================================================
        for (int j = 0; j < num_nu; ++j) {
            //int nu = j + 1;
            double integral_J = 0.0;
            for (int k = 0; k < num_h; ++k) {
                double integral = 0.0;
                #pragma unroll 4
                for (int m = 0; m <= (i-1); ++m) {
                    int tau_index = i - m; 
                    integral += g_table_vals(j,k,tau_index) * s_vals(k, m) * sigma_vals(j, k) * J_vals(j, m);
                }
                integral *= dt;
                integral_J += tilde_sigma_vals(j, k) * integral;
            }
            J_vals(j, i) = R_vals(j,i) * integral_J + R_vals(j,i) * IntgImpG_vals(j, i);
        }

        // ====================================================================
        // STEP 2: Compute H_vals[j, k, i] (Recovery/Removal Flux)
        // ====================================================================
        // Equation: H[j,k,i] = SUM_{m=0}^{i-1} s[m] * sigma[j,k] * J[j,m] * gamma[tau_{i-m}] * dt
        //                      + F_gamma[j,k,i]
        // ====================================================================
        for (int j = 0; j < num_nu; ++j) {
            for (int k = 0; k < num_h; ++k) {
                if (sigma_vals(j, k) != 0) {
                    double integral_H = 0.0;
                    #pragma unroll 4
                    for (int m = 0; m <= (i-1); ++m) {
                         int tau_index = i - m  ; 
                        integral_H += s_vals(k, m ) * sigma_vals(j, k) * J_vals(j, m ) * gamma_table_vals(j,k,tau_index) ;
                    }
                    integral_H *= dt;
                    H_vals(j, k, i) = integral_H + IntgImpGamma_vals(j, k, i);
                }
            }
        }

        // ====================================================================
        // STEP 3: Update s_vals[k, i] (Susceptibility)
        // ====================================================================
        // Equation: s[k,i] = s[k,i-1] - dt*(I_h - sum_H1 - sum_H2)
        //           where I_h = SUM_j s[i-1] * sigma[j,k] * J[j,i-1]
        // ====================================================================
        for (int k = 0; k < num_h; ++k) {
            const set<int>& h = h_values_all[k];
            double I_h = 0.0;
            
            // Compute infection incidence at previous time step i-1
            for (int j = 0; j < num_nu; ++j) {
                if (i>1){
                    I_h += s_vals(k, i - 1) * sigma_vals(j, k) * J_vals(j, i - 1);    
                }    
            }

            double sum_H1 = 0.0, sum_H2 = 0.0;
            for (int u : h) {
                int nu_index = u - 1;
                sum_H1 += H_vals(nu_index, h_index_map[h], i - 1);

                set<int> h_minus_u = h;
                h_minus_u.erase(u);
                sum_H2 += H_vals(nu_index, h_index_map[h_minus_u], i - 1);

            }
            s_vals(k, i) = s_vals(k, i - 1) - dt * (I_h - sum_H1 - sum_H2);
        }
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    //std::cout << "Execution time : " << elapsed.count() << " seconds\n";
    std::cout.flush();

}


void time_evolution_uncoupled(
    py::array_t<double> t_values, 
    py::array_t<double> J_values, 
    py::array_t<double> s_values,
    py::array_t<double> R_nu,
    py::array_t<double> sigma_nu_h,
    py::array_t<double> tilde_sigma_nu_h,
    py::array_t<double> g_table, 
    py::array_t<double> gamma_table, 
    py::array_t<double> IntgImpG,
    py::array_t<double> IntgImpGamma,
    vector<int> nu_values,
    vector<set<int>> h_values_all,  
    vector<set<int>> h_values_appx,
    const py::dict& h_index_map_py  // Accept dict[tuple, int]
) {
    auto start = std::chrono::high_resolution_clock::now();
    auto t_vals = t_values.unchecked<1>();
    auto J_vals = J_values.mutable_unchecked<2>();
    auto s_vals = s_values.mutable_unchecked<2>();
    auto R_vals = R_nu.unchecked<2>();
    auto sigma_vals = sigma_nu_h.unchecked<2>();
    auto tilde_sigma_vals = tilde_sigma_nu_h.unchecked<2>();
    auto g_table_vals = g_table.mutable_unchecked<3>();
    auto gamma_table_vals = gamma_table.mutable_unchecked<3>();
    auto IntgImpG_vals = IntgImpG.mutable_unchecked<2>();
    auto IntgImpGamma_vals = IntgImpGamma.mutable_unchecked<3>();

    std::map<std::set<int>, int> h_index_map = convert_h_index_map(h_index_map_py);

    int n_steps = t_vals.shape(0);
    int num_nu = J_vals.shape(0);
    int num_h = h_values_all.size();
    int num_h_appx = h_values_appx.size();
    double dt = t_vals(1) - t_vals(0);

    //std::vector<std::vector<double>> H_vals(num_nu, std::vector<double>(num_h));
    //py::array_t<double> H_val = py::array::zeros({num_nu, num_h, n_steps}, py::dtype::of<double>());
    //auto H_val = py::array_t<double>::zeros({num_nu, num_h, n_steps});
    //py::array_t<double> H_values = py::array_t<double>::zeros({num_nu, num_h, n_steps});
    py::array_t<double> H_values({num_nu, num_h, n_steps});
    auto buf = H_values.request();
    std::memset(buf.ptr, 0, buf.size * sizeof(double));
    auto H_vals = H_values.mutable_unchecked<3>();

    vector<int> hindex(num_h_appx);
    for (int k = 0; k < num_h_appx; ++k) {
        const set<int>& h = h_values_appx[k];
        hindex[k] = h_index_map[h];
        //std::cout << "h index " << k << " : "<<hindex[k] << " ; ";
            }

    for (int i = 1; i < n_steps; ++i) {
        double t = t_vals(i);

        // ====================================================================
        // STEP 1: Compute J (SKIPPED in uncoupled - only for approximated histories)
        // In the uncoupled version, J is not recomputed, it uses the provided values
        // This is correct for the approximation (subset of histories)
        // =====================================================================
        
        // ====================================================================
        // STEP 2: Compute H_vals[j, k, i] for ALL histories
        // ====================================================================
        for (int j = 0; j < num_nu; ++j) {
            for(int k = 0; k < num_h; ++k){
                if (sigma_vals(j, k) != 0) {
                    double integral_H = 0.0;
                    #pragma unroll 4
                    for (int m = 0; m <= (i-1); ++m) {
                         int tau_index = i - m ; 
                        integral_H += s_vals(k, m ) * sigma_vals(j, k) * J_vals(j, m )*gamma_table_vals(j,k,tau_index) ;
                    }
                    integral_H *= dt;
                    H_vals(j, k, i) = integral_H + IntgImpGamma_vals(j, k, i);
                }
            }
        }
        
        // ====================================================================
        // STEP 3: Update s_vals[k, i] for APPROXIMATED histories only
        // ====================================================================
        for (int k1 = 0; k1 < num_h_appx; ++k1) {
            const set<int>& h = h_values_appx[k1];
            int k = hindex[k1];
            double I_h = 0.0;
            // Compute infection incidence at previous time step
            for (int j = 0; j < num_nu; ++j) {
                if (i>1){
                    I_h += s_vals(k, i - 1) * sigma_vals(j, k) * J_vals(j, i - 1);    
                } 
            }

            double sum_H1 = 0.0, sum_H2 = 0.0;
            for (int u : h) {
                int nu_index = u - 1;
                //sum_H1 += H_vals[nu_index][h_index_map[h]];
                sum_H1 += H_vals(nu_index, h_index_map[h], i - 1);
                set<int> h_minus_u = h;
                h_minus_u.erase(u);
                //sum_H2 += H_vals[nu_index][h_index_map[h_minus_u]];
                sum_H2 += H_vals(nu_index, h_index_map[h_minus_u], i - 1);

            }
            s_vals(k, i) = s_vals(k, i - 1) - dt * (I_h - sum_H1 - sum_H2);
        }
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    //std::cout << "Execution time : " << elapsed.count() << " seconds\n";
    std::cout.flush();

}

PYBIND11_MODULE(infectionRenewalEq_x86_v3, m) {
    m.def("time_evolution_coupled", &time_evolution_coupled, "Run time evolution coupled sets",
        py::arg("t_values"), 
        py::arg("J_values"),
        py::arg("s_values"),
        py::arg("R_nu"), 
        py::arg("sigma_nu_h"),
        py::arg("tilde_sigma_nu_h"),
        py::arg("g_table"),
        py::arg("gamma_table"),
        py::arg("IntgImpG"),
        py::arg("IntgImpGamma"),
        py::arg("nu_values"), 
        py::arg("h_values_all"),
        py::arg("h_index_map_py"));

    m.def("time_evolution_uncoupled", &time_evolution_uncoupled, "Run time evolution uncoupled",
        py::arg("t_values"), 
        py::arg("J_values"),
        py::arg("s_values"),
        py::arg("R_nu"), 
        py::arg("sigma_nu_h"),
        py::arg("tilde_sigma_nu_h"),
        py::arg("g_table"),
        py::arg("gamma_table"),
        py::arg("IntgImpG"),
        py::arg("IntgImpGamma"),
        py::arg("nu_values"), 
        py::arg("h_values_all"),
        py::arg("h_values_appx"),
        py::arg("h_index_map_py"));
}
