#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <fftw3-mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _DAM_DEBUG
#include <sys/types.h>
#include <unistd.h>
#endif //_DAM_DEBUG

#include "common.h"

//#define LOGTIMESTEP 1

static int mpi_init(int* p_argc,char ***p_argv); //DAM used
static void fft_init(int threads_ok); //DAM used

int main(int argc,char* argv[])
{
  const int multi_thread=mpi_init(&argc,&argv);
  msg_init();
  timer_set_category(Init);

  //
  // Initialization / Memory allocation
  //						      
  if(argc<2)
    msg_abort(1,"Error: Parameter file not specified. cola_halo param.init\n");
  read_parameters(argv[1]);

  msg_set_loglevel(Param.loglevel);

  fft_init(multi_thread);
  comm_init(Param.nc,Param.boxsize);

  cosmo_init(Param.power_spectrum_filename,Param.sigma8,
	     Param.omega_m,1-Param.omega_m);

  const double a_init=Param.a_final;

  lpt_init(Param.nc);

  MPI_Barrier(MPI_COMM_WORLD);
  int seed=Param.random_seed;

  // Sets initial grid and 2LPT displacement
  timer_set_category(LPT);
  lpt_set_displacement(seed,Param.boxsize);

  //Writes initial condition
  msg_printf(info,
	     "Writing initial condition to file for e.g. Gadget N-body simulation\n");
  char filename[256];
  sprintf(filename,"%s",Param.init_filename);
  if(Param.nbox_per_side<=0)
    write_snapshot(filename,a_init);
  else
    write_snapshot_cola(filename,a_init);

  timer_print();

  lpt_end();

  MPI_Finalize();
  return 0;
}

int mpi_init(int* p_argc,char*** p_argv)
{
  // MPI+OpenMP paralellization: MPI_THREAD_FUNNELED
  // supported by mpich2 1.4.1, but now by openmpi 1.2.8

#ifdef _OPENMP
  int thread_level, hybrid_parallel;
  MPI_Init_thread(p_argc,p_argv,MPI_THREAD_FUNNELED,&thread_level);
  hybrid_parallel=(thread_level>=MPI_THREAD_FUNNELED);

  int myrank; MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  if(myrank==0) {
    if(hybrid_parallel)
      printf("MPI + multi thread supported (MPI_THREAD_FUNNELED).\n");
    else
      printf("Warning: MPI + multi thread not supported. 1 thread per node.\n");
  }
	
  return hybrid_parallel;
#else
  MPI_Init(p_argc,p_argv);
  int myrank; MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0)
    printf("MPI only without OpenMP\n");
  return 0;
#endif

}

void fft_init(int threads_ok)
{
  // Initialize FFTW3

#ifdef _OPENMP
  if(threads_ok)
    threads_ok=fftwf_init_threads();
  if(!threads_ok)
    msg_printf(warn,"Multi-thread FFTW not supported.\n");
#endif
    
  fftwf_mpi_init();

#ifdef _OPENMP
  if(threads_ok) {
    int nthreads=omp_get_max_threads();
    fftwf_plan_with_nthreads(nthreads);
    msg_printf(info,"Multi-threaded FFTW: %d threads\n",nthreads);
  }
#endif

}
