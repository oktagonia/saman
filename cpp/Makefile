CXX = g++
CXXFLAGS = -Wall -std=c++20 -O2 -I./src \
           -I/opt/homebrew/include/eigen3 \
           -I/usr/local/include/eigen3

all: ellipse torus

ellipse: example/ellipse.cpp src/manifold.hpp
	$(CXX) $(CXXFLAGS) example/ellipse.cpp -o ellipse

torus: example/torus.cpp src/manifold.hpp
	$(CXX) $(CXXFLAGS) example/torus.cpp -o torus

python: bindings/python/module.cpp src/manifold.hpp
	$(CXX) $(CXXFLAGS) \
		-shared \
		-fPIC `python3 -m pybind11 --includes` \
		bindings/python/module.cpp \
		-undefined dynamic_lookup \
		-o ../saman`python3-config --extension-suffix`

clean:
	rm -f ellipse torus

