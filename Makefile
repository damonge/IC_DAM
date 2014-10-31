#
# IC_DAM
#   

CC      = mpicc -std=c99 
WOPT    ?= -Wall
CFLAGS  := -O3 $(WOPT)
CFLAGS += #-D_LONGIDS #-D_DAM_SAVEMEM
LIBS    := -lm

# Define paths of FFTW3 & GSL libraries if necessary.
FFTW3_DIR ?= /home/damonge
GSL_DIR   ?= 

DIR_PATH = $(FFTW3_DIR) $(GSL_DIR)

CFLAGS += $(foreach dir, $(DIR_PATH), -I$(dir)/include)
LIBS   += $(foreach dir, $(DIR_PATH), -L$(dir)/lib)

EXEC = IC_DAM
all: $(EXEC)

OBJS := src/main.o
OBJS += src/read_param.o src/lpt.o src/msg.o src/cosmo.o
OBJS += src/comm.o src/timer.o

LIBS += -ldl
LIBS += -lgsl -lgslcblas
LIBS += -lfftw3f_mpi -lfftw3f


IC_DAM: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LIBS) -o $@

src/main.o: src/main.c src/common.h
src/comm.o: src/comm.c src/common.h
src/lpt.o: src/lpt.c src/common.h
src/msg.o: src/msg.c src/common.h
src/cosmo.o: src/cosmo.c src/common.h
src/read_param.o: src/read_param.c src/common.h
src/timer.o: src/timer.c src/common.h

clean :
	rm -f $(EXEC) $(OBJS)

cleaner :
	rm -f $(EXEC) $(OBJS) *~ src/*~
