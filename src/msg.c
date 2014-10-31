//
// Utility functions to write message, record computation time ...
//

#include <stdio.h>
#include <stdarg.h>
#include <mpi.h>

#include "common.h"

static int myrank=-1;

// Initialize using msg_ functions.
void msg_init()
{
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
}

void msg_printf(const char *fmt, ...)
{
  if(myrank==0) {
    va_list argp;

    va_start(argp,fmt);
    vfprintf(stdout,fmt,argp);
    fflush(stdout);
    va_end(argp);
  }
}

void msg_abort(const int errret,const char *fmt, ...)
{
  va_list argp;

  va_start(argp,fmt);
  vfprintf(stderr,fmt,argp);
  va_end(argp);

  MPI_Abort(MPI_COMM_WORLD,errret);
}  
