CXX = g++
CXXFLAGS = -std=c++20 -O2 -I./src \
           -I/opt/homebrew/include/eigen3 \
           -I/usr/local/include/eigen3

# default target
all: ellipse

ellipse: example/ellipse.cpp src/manifold.hpp
	$(CXX) $(CXXFLAGS) example/ellipse.cpp -o ellipse

clean:
	rm -f ellipse

