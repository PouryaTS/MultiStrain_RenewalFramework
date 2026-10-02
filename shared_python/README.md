# Shared Python Notebooks

This folder contains selected notebooks for simulating multi-strain infection dynamics with renewal equations and analyzing the results. `solve_renewal_equations.ipynb` runs the solvers and parameter sweeps; `CreatErrorHeatmaps.ipynb` visualizes model errors.

The simulation notebook imports compiled C++ modules from `../Code_main/` and reads saved results from `../Data/`. The heatmap notebook reads the saved error array from that data folder. Generated data and figures use `../data/` and `../figures/`.

Keep only the Python code you intend to share here. The original `Python Scripts/` folder is ignored by Git.
