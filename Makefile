all: libfoo.so run
libfoo.so: triton-aoti.py
	python3 triton-aoti.py
run: libfoo.so run.cpp
	g++ -g -o run run.cpp -I /usr/local/cuda/include -L. -lcudart -L/usr/local/cuda/lib64 -ldl -rdynamic -lfoo
test: libfoo.so run
	LD_LIBRARY_PATH=$(shell pwd) LD_PRELOAD=libfoo.so ./run
clean:
	rm -f run libfoo.so
