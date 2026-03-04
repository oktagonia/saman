CXX ?= g++
PYTHON ?= python3

UNAME_S := $(shell uname -s)

CXXFLAGS = -Wall -std=c++20 -O2 -I./ \
           -I/opt/homebrew/include/eigen3 \
           -I/usr/local/include/eigen3 \
           -I/usr/include/eigen3

PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")

ifeq ($(UNAME_S),Darwin)
PY_MODULE_LDFLAGS := -undefined dynamic_lookup
else
PY_MODULE_LDFLAGS :=
endif

.PHONY: all python clean

all: ellipse torus

ellipse: example/ellipse.cpp manifold.hpp
	$(CXX) $(CXXFLAGS) example/ellipse.cpp -o ellipse

torus: example/torus.cpp manifold.hpp
	$(CXX) $(CXXFLAGS) example/torus.cpp -o torus

python: python/module.cpp manifold.hpp
	$(CXX) $(CXXFLAGS) \
		-shared \
		-fPIC $(PYBIND11_INCLUDES) \
		python/module.cpp \
		$(PY_MODULE_LDFLAGS) \
		-o ../saman$(EXT_SUFFIX)

clean:
	rm -f ellipse torus

