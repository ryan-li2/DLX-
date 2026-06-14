CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -pthread

GUROBI_HOME ?= /Library/gurobi1302/macos_universal2
GUROBI_LIB ?= gurobi130
ORTOOLS_HOME ?= $(HOME)/Downloads/or-tools_arm64_macOS-15.3.1_cpp_v9.12.4544

.PHONY: all dlx gurobi cpsat clean

all: dlx

dlx: dlx_experiment

dlx_experiment: dlx_experiment.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

gurobi: gurobi_experiment

gurobi_experiment: dlx_experiment.cpp gurobi_spk.cpp gurobi_spk.hpp
	$(CXX) $(CXXFLAGS) -DUSE_GUROBI \
		-I"$(GUROBI_HOME)/include" \
		-L"$(GUROBI_HOME)/lib" \
		dlx_experiment.cpp gurobi_spk.cpp \
		-lgurobi_c++ -l$(GUROBI_LIB) \
		-Wl,-rpath,"$(GUROBI_HOME)/lib" \
		-o $@

cpsat: cpsat_experiment

cpsat_experiment: dlx_experiment.cpp cpsat_spk.cpp cpsat_spk.hpp CMakeLists.txt
	cmake -S . -B build/cpsat \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_PREFIX_PATH="$(ORTOOLS_HOME)" \
		-DBUILD_CPSAT=ON
	cmake --build build/cpsat --config Release

clean:
	rm -f dlx_experiment gurobi_experiment cpsat_experiment
	rm -rf build
