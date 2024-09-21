all: foo.so run
foo.so: triton-aoti.py
	python3 triton-aoti.py
run: run.cpp
	g++ -g -o run run.cpp -ldl -rdynamic
test: foo.so run
	./run
clean:
	rm -f run foo.so
