//
// MPI Communication
//
// #include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <fftw3-mpi.h>

#include "common.h"

static int ThisNode,NNode;
static int Tag;

static float BoxSize;

void comm_init(const int nc_p,const float boxsize)
{
  msg_printf(verbose, "comm initialization\n");

  MPI_Comm_rank(MPI_COMM_WORLD, &ThisNode);
  MPI_Comm_size(MPI_COMM_WORLD, &NNode);

  ptrdiff_t local_nx,local_x_start;
  int* local_x_table=malloc(sizeof(int)*NNode*2); assert(local_x_table);
  int* local_nx_table=local_x_table+NNode;

  // Print LPT initial domain decomposition (just for information**)
  fftwf_mpi_local_size_3d(nc_p,nc_p,nc_p,MPI_COMM_WORLD,
			  &local_nx,&local_x_start);

  MPI_Allgather(&local_nx,1,MPI_INT,local_nx_table,1,MPI_INT, 
		MPI_COMM_WORLD);
  MPI_Allgather(&local_x_start,1,MPI_INT,local_x_table,1,MPI_INT, 
		MPI_COMM_WORLD);

  for(int i=0;i<NNode;i++)
    msg_printf(debug,"LPT Task=%d x=%d..%d\n",i,local_x_table[i],
	       local_x_table[i]+local_nx_table[i]-1);

  BoxSize=boxsize;
  Tag=600;

  free(local_x_table);
}

int comm_this_node(void)
{
  return ThisNode;
}

int comm_nnode(void)
{
  return NNode;
}
