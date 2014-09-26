#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <fftw3-mpi.h>

#include "common.h"

Snapshot *allocate_snapshot(const int nc,const int nx,const double np_alloc_factor,
			    void* const mem,const size_t mem_size)
{
  Snapshot* snapshot=malloc(sizeof(Snapshot));

  snapshot->np_allocated=(int)(np_alloc_factor*nc*nc*(nx+1));
  long long nc_long=nc;
  snapshot->np_total=nc_long*nc_long*nc_long;
  snapshot->p=mem;
  assert(mem_size>=sizeof(ParticleMinimum)*(snapshot->np_allocated));
  snapshot->nc=nc;
  snapshot->a=0.0f; //snapshot->a_v= 0.0f; snapshot->a_x= 0.0f;

  return snapshot;
}

void allocate_shared_memory(const int nc,const double np_alloc_factor,
			    Memory* const mem)
{
  // Allocate shared memory

  // mem1
  //  2LPT grids / PM density grid

  // Memory for 2LPT (6*np_local words)
  ptrdiff_t local_nx,local_x_start;
  ptrdiff_t size_lpt_one=fftwf_mpi_local_size_3d(nc,nc,nc,MPI_COMM_WORLD,
						 &local_nx,&local_x_start);
#ifdef _DAM_SAVEMEM
  size_lpt_one/=nc;
  size_lpt_one*=(nc/2+1);
#endif //_DAM_SAVEMEM
  ptrdiff_t ncomplex_lpt=12*size_lpt_one; //DAM: why 12 and not 9?

  const int np_alloc=(int)(np_alloc_factor*nc*nc*(local_nx+1));

  msg_printf(verbose,"%d Mbytes requested for LPT\n",
	     (int)(ncomplex_lpt*sizeof(fftwf_complex)/(1024*1024)));
  
  ptrdiff_t ncomplex1= ncomplex_lpt;

  mem->mem1=fftwf_alloc_complex(ncomplex1);
  mem->size1=sizeof(fftwf_complex)*ncomplex1;

  if(mem->mem1==0)
    msg_abort(0050,"Error: Unable to allocate %d Mbytes for mem1\n",
	      (int)(mem->size1/(1024*1024)));
  
  // mem2
  // PM density_k mesh and snapshot
  size_t size_snapshot=sizeof(ParticleMinimum)*np_alloc;
  size_t ncomplex2=size_snapshot/sizeof(fftwf_complex)+1;
  msg_printf(verbose,"%d Mbytes requested for snapshot in mem2\n",
	     (int)(size_snapshot/(1024*1024)));

  mem->mem2=fftwf_alloc_complex(ncomplex2);
  mem->size2=sizeof(fftwf_complex)*ncomplex2;

  if(mem->mem2==0)
    msg_abort(0060,"Error: Unable to allocate %d + %d Mbytes for mem1&2.\n",
	      (int)(mem->size1/(1024*1024)),(int)(mem->size2/(1024*1024)));

  msg_printf(info,"%d Mbytes allocated for mem1.\n", 
	     (int)(mem->size1/(1024*1024)));
  msg_printf(info,"%d Mbytes allocated for mem2.\n", 
	     (int)(mem->size2/(1024*1024)));
}
