# Build the Python Extensions

The C++ files in this directory use pybind11 to build two Python extension modules:

- `infectionRenewalEq_x86_v3.cpp` builds `infectionRenewalEq_x86_v3`.
- `time_evolution_seqcirc_approx.cpp` builds `infectionRenewalEq_seqcirc_approx`.

Build the modules with the same Python environment that will import them. The commands below are for macOS and require a C++ compiler, Python development headers, and `pybind11` installed in the active environment.

```sh
cd Code_main
python3 -m pip install pybind11
which python3
which python3-config
python3 -c "import platform; print(platform.machine())"
```

The `python3` and `python3-config` commands must refer to the same environment. The architecture check should print `x86_64` for the x86 build or the machine's native architecture for the native build.

## x86_64 Build

Activate the x86_64 environment first, for example `conda activate py_x86`, then run both commands from `Code_main/`:

```sh
arch -x86_64 c++ -O3 -march=native -ffast-math -funroll-loops -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) infectionRenewalEq_x86_v3.cpp \
    -o "infectionRenewalEq_x86_v3$(python3-config --extension-suffix)" \
    -undefined dynamic_lookup

arch -x86_64 c++ -O3 -march=native -ffast-math -funroll-loops -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) time_evolution_seqcirc_approx.cpp \
    -o "infectionRenewalEq_seqcirc_approx$(python3-config --extension-suffix)" \
    -undefined dynamic_lookup
```

## Native macOS Build

Use a Python environment running natively on the machine, then run:

```sh
c++ -O3 -march=native -ffast-math -funroll-loops -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) infectionRenewalEq_x86_v3.cpp \
    -o "infectionRenewalEq_x86_v3$(python3-config --extension-suffix)" \
    -undefined dynamic_lookup

c++ -O3 -march=native -ffast-math -funroll-loops -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) time_evolution_seqcirc_approx.cpp \
    -o "infectionRenewalEq_seqcirc_approx$(python3-config --extension-suffix)" \
    -undefined dynamic_lookup
```

`-ffast-math` enables faster floating-point transformations that can reduce numerical precision. `-march=native` tunes the binaries for the build machine, so rebuild the modules for each Python version and target architecture instead of reusing compiled `.so` files across environments.

Check that both modules import from the active environment:

```sh
python3 -c "import infectionRenewalEq_x86_v3, infectionRenewalEq_seqcirc_approx; print('Both modules imported')"
```
