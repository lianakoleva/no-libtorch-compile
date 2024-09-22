all: libfoo.so run
libfoo.so: triton-aoti.py
	python3 triton-aoti.py
run: libfoo.so run.cpp
	g++ -g -o run run.cpp -L. -ldl -rdynamic -lfoo
test: libfoo.so run
	./run
clean:
	rm -f run libfoo.so
