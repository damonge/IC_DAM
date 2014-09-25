#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <fftw3-mpi.h>

#include "common.h"

Particles *allocate_particles(const int nc,const int nx,const double np_alloc_factor)
{
  Particles *particles=malloc(sizeof(Particles));

  const int np_alloc=(int)(np_alloc_factor*nc*nc*(nx+1));

  particles->p=malloc(sizeof(Particle)*np_alloc);
  if(particles->p==0)
    msg_abort(0010,"Error: Failed to allocate memory for particles\n");

  particles->force=malloc(sizeof(float)*3*np_alloc);
  if(particles->force==0)
    msg_abort(0010,"Error: Failed to allocate memory for particle forces\n");

  msg_printf(info,"%d Mbytes allocated for %d particles (alloc_factor= %.2lf)\n",
	     (sizeof(Particle)+3*sizeof(float))*np_alloc/(1024*1024),
	     np_alloc,np_alloc_factor);

  particles->np_allocated=np_alloc;

  particles->np_total=(long long)nc*nc*nc;
  
  const int nnode=comm_nnode();
  particles->np_average=(float)(pow((double)nc,3)/nnode);

  return particles;
}

#ifdef _LIGHTCONE
Snapshot *allocate_lightcone(const int nc,const int nx,const double np_alloc_factor)
{
  Snapshot *snap=malloc(sizeof(Snapshot));
  const int nnode=comm_nnode();
  const int np_alloc=(int)(np_alloc_factor*nc*nc*(nx+1));

  snap->p=malloc(sizeof(ParticleMinimum)*np_alloc);
  if(snap->p==0)
    msg_abort(0010,"Error: Failed to allocate memory for lightcone snapshot\n");
  msg_printf(info,"%d Mbytes allocated for lightcone\n",
	     sizeof(ParticleMinimum)*np_alloc/(1024*1024));

  long long nc_long=nc;
  snap->np_local=0;
  snap->np_allocated=np_alloc;
  snap->np_total=nc_long*nc_long*nc_long;
  snap->np_average=(float)(pow((double)nc,3)/nnode);
  snap->a=-1.0;
  snap->nc=nc;

  return snap;
}
#endif //_LIGHTCONE

Snapshot *allocate_snapshot(const int nc,const int nx,const int np_alloc,
			    void* const mem,const size_t mem_size)
{
  Snapshot* snapshot=malloc(sizeof(Snapshot));

  snapshot->np_allocated=np_alloc;
  long long nc_long=nc;
  snapshot->np_total=nc_long*nc_long*nc_long;
  snapshot->p=mem; assert(mem_size>=sizeof(ParticleMinimum)*np_alloc);
  snapshot->nc=nc;
  snapshot->a=0.0f; //snapshot->a_v= 0.0f; snapshot->a_x= 0.0f;

  return snapshot;
}

void allocate_shared_memory(const int nc,const int nc_factor,
			    const double np_alloc_factor,Memory* const mem)
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
  
  // Memory for PM (nc_factor^3 * np_local each)
  const int Ngrid=nc_factor*nc;
  ptrdiff_t local_ny,local_y_start;
  ptrdiff_t size_pm_one=
    fftwf_mpi_local_size_3d_transposed(Ngrid,Ngrid,Ngrid,MPI_COMM_WORLD,
				       &local_nx,&local_x_start,&local_ny,&local_y_start);
#ifdef _DAM_SAVEMEM
  size_pm_one/=Ngrid;
  size_pm_one*=(Ngrid/2+1);
#endif //_DAM_SAVEMEM
  ptrdiff_t ncomplex_pm=size_pm_one;

  msg_printf(verbose,"%d Mbytes requested for one PM grid\n",
	     (int)(ncomplex_pm*sizeof(fftwf_complex)/(1024*1024)));

  msg_printf(verbose,"PM size %d %d %d\n",size_pm_one,
	     local_nx*Ngrid*Ngrid,local_nx);

  ptrdiff_t ncomplex1= ncomplex_lpt > ncomplex_pm ? ncomplex_lpt : ncomplex_pm;

  mem->mem1=fftwf_alloc_complex(ncomplex1);
  mem->size1=sizeof(fftwf_complex)*ncomplex1;

  if(mem->mem1==0)
    msg_abort(0050,"Error: Unable to allocate %d Mbytes for mem1\n",
	      (int)(mem->size1/(1024*1024)));
  
  // mem2
  // PM density_k mesh and snapshot
  size_t ncomplex2=(Ngrid/2+1)*Ngrid*local_ny; //ncomplex_pm;
  size_t size2=sizeof(fftwf_complex)*(Ngrid/2+1)*Ngrid*local_ny;

  msg_printf(verbose,"%d Mbytes requested for delta_k mesh (mem2). ny=%d\n",
	     (int)(size2/(1024*1024)),local_ny);

  size_t size_snapshot=sizeof(ParticleMinimum)*np_alloc;
  msg_printf(verbose,"%d Mbytes requested for snapshot in mem2\n",
	     (int)(size_snapshot/(1024*1024)));
  if(size_snapshot>size2) {
    size2=size_snapshot;
    ncomplex2=size_snapshot/sizeof(fftwf_complex)+1;
  }

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
