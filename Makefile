

all: lib/var_model_acc.so

INCDIRS=-I/usr/include/python2.6 -I$(HOME)/.local/lib/python2.6/site-packages/numpy/core/include

lib/var_model_acc.so: lib/var_model_acc.c
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing $(INCDIRS) -o lib/var_model_acc.so lib/var_model_acc.c
	rm lib/var_model_acc.c

lib/var_model_acc.c: src/var_model_acc.pyx
	cython -o lib/var_model_acc.c src/var_model_acc.pyx

	
clean:
	rm lib/var_model_acc.so
