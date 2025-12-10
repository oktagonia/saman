CXX = g++
CXXFLAGS = -Wall -std=c++20 -O2 -I./ \
           -I/opt/homebrew/include/eigen3 \
           -I/usr/local/include/eigen3

all: ellipse torus

ellipse: example/ellipse.cpp manifold.hpp
	$(CXX) $(CXXFLAGS) example/ellipse.cpp -o ellipse

torus: example/torus.cpp manifold.hpp
	$(CXX) $(CXXFLAGS) example/torus.cpp -o torus

python: python/module.cpp manifold.hpp
	$(CXX) $(CXXFLAGS) \
		-shared \
		-fPIC `python3 -m pybind11 --includes` \
		python/module.cpp \
		-undefined dynamic_lookup \
		-o ../saman`python3-config --extension-suffix`

clean:
	rm -f ellipse torus

