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

int mpi_init(int* p_argc,char ***p_argv);
void fft_init(int threads_ok);
#ifdef _LIGHTCONE
void output_lightcone(Snapshot * const lightcone,Memory mem);
#endif //_LIGHTCONE
void snapshot_time(const float aout,const int iout, 
		   Particles const * const particles, 
		   Snapshot * const snapshot,
		   Memory mem,int do_lpt);
void get_mem(void)
{
#ifdef _DAM_DEBUG
  int npag;
  double memo;
  char fname_mem[256];
  FILE *fil;

  sprintf(fname_mem,"/proc/%d/statm",getpid());
  fil=fopen(fname_mem,"r");
  fscanf(fil,"%d",&npag);
  fclose(fil);
  memo=npag*getpagesize()/(1024*1024);

  printf("Using %lf MB\n",memo);
  scanf("%d",&npag);
#endif //_DAM_DEBUG
}

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
  comm_init(Param.pm_nc_factor*Param.nc,Param.nc,Param.boxsize);


  cosmo_init(Param.power_spectrum_filename,Param.sigma8,
	     Param.omega_m,1-Param.omega_m);

  set_a_final();

#ifndef LOGTIMESTEP
  const double da=Param.a_final/(Param.ntimestep+0);
  const double a_init=da;
#else
  const double a_init=0.1;
#endif

  Memory mem; 
  allocate_shared_memory(Param.nc,Param.pm_nc_factor,Param.np_alloc_factor,&mem);

  lpt_init(Param.nc,mem.mem1,mem.size1);
  const int local_nx=lpt_get_local_nx();

  Particles* particles=allocate_particles(Param.nc,local_nx,Param.np_alloc_factor);
#ifdef _LIGHTCONE
  Snapshot* lightcone=allocate_lightcone(Param.nc,local_nx,Param.np_alloc_factor);
#endif //_LIGHTCONE
  Snapshot* snapshot= allocate_snapshot(Param.nc,local_nx,
					particles->np_allocated,mem.mem2,mem.size2);
  
  pm_init(Param.pm_nc_factor*Param.nc,Param.pm_nc_factor,Param.boxsize,
	  mem.mem1,mem.size1,mem.mem2,mem.size2);

#ifndef _LIGHTCONE
  for(int i=0;i<Param.n_aout;i++) {
    msg_printf(verbose,"zout[%d]= %lf, aout= %f\n", 
	       i,1.0/Param.aout[i]-1,Param.aout[i]);
  }
#endif //_LIGHTCONE

  MPI_Barrier(MPI_COMM_WORLD);
  int seed=Param.random_seed;
#ifndef _LIGHTCONE
  int iout=0;
#endif //_LIGHTCONE

  // Sets initial grid and 2LPT displacement
  timer_set_category(LPT);
  lpt_set_displacement(seed,Param.boxsize,particles);
  if(Param.write_disp) {
    msg_printf(info,"Writing displacement field\n");
    char filename[256];
    sprintf(filename,"%s",Param.disp_filename);
    write_displacements(filename,particles);
  }

  //Writes initial condition
  if(Param.write_init) {
    msg_printf(info,
	       "Writing initial condition to file for e.g. Gadget N-body simulation\n");
    char filename[256];
    sprintf(filename,"%s",Param.init_filename);
    cola_set_LPT_snapshot(a_init,particles,snapshot);
    write_snapshot(filename,snapshot);
  }

#ifndef _LIGHTCONE
  //Writes snapshots prior to a_init
  while(iout<Param.n_aout && Param.aout[iout]<a_init) {
    msg_printf(info,"a = %lf is before initial condition. Writing as 2LPT\n",
	       Param.aout[iout]);
    snapshot_time(Param.aout[iout],iout,particles,snapshot,mem,1);
    iout++;
  }
  if(iout==Param.n_aout) {
    timer_print();
    MPI_Finalize();
    return 0;
  }
#endif //_LIGHTCONE

  //Sets initial condition
  lpt_set_initial_condition(a_init,Param.boxsize,particles);

#ifdef _LIGHTCONE
  cola_add_to_lightcone_LPT(a_init,particles,lightcone);
#endif //_LIGHTCONE

  timer_set_category(COLA);

#ifndef LOGTIMESTEP
  msg_printf(info,"timestep linear in a\n");
#else
  const double loga_init=log(a_init);
  const double dloga=(log(Param.a_final)-log(a_init))/Param.ntimestep;
  particles->a_v=exp(log(a_init)-0.5*dloga);
  msg_printf(info,"timestep linear in loga\n");
#endif

  get_mem();

  //
  // Time evolution loop
  //
#ifdef _LIGHTCONE
  int n_aout=1;
#else //_LIGHTCONE
  int n_aout=Param.n_aout;
#endif //_LIGHTCONE
  if(n_aout>0 && Param.ntimestep>1 && Param.a_final>a_init) {
    msg_printf(normal,"Time integration a= %g -> %g, %d steps\n", 
	       a_init,Param.a_final,Param.ntimestep);
    for (int istep=1;istep<=Param.ntimestep;istep++) {
      msg_printf(normal,"Timestep %d/%d\n",istep,Param.ntimestep);
      
      timer_start(comm);
      // move particles to other nodes
      move_particles2(particles,Param.boxsize,mem.mem1,mem.size1);
      timer_stop(comm);
      pm_calculate_forces(particles);

#ifndef LOGTIMESTEP
      double avel0=particles->a_v;
      double apos0=particles->a_x;
      
      double avel1=(istep+0.5)*da;
      double apos1=(istep+1.0)*da;
#else
      float avel0=exp(loga_init + (istep-0.5)*dloga);
      float apos0=exp(loga_init + istep*dloga);
      
      float avel1=exp(loga_init+(istep+0.5)*dloga);
      float apos1=exp(loga_init+(istep+1)*dloga);
#endif
      if(avel1<=avel0 || apos1<=apos0 || apos1<=avel1)
	msg_abort(2,"Something went wrong...\n");

#ifdef _LIGHTCONE
      cola_add_to_lightcone(particles,lightcone);
#else //_LIGHTCONE
      while(iout<Param.n_aout && avel0<=Param.aout[iout] && Param.aout[iout]<=apos0) {
	snapshot_time(Param.aout[iout],iout,particles,snapshot,mem,0);
	iout++;
      }
      if(iout>=Param.n_aout) break;
#endif //_LIGHTCONE
      if(avel1>Param.a_final) break;

      // Leap-frog "kick" -- velocities updated
      cola_kick(particles,avel1);

#ifdef _LIGHTCONE
      cola_add_to_lightcone(particles,lightcone);
#else //_LIGHTCONE
      while(iout<Param.n_aout && apos0<Param.aout[iout] && Param.aout[iout]<=avel1) {
	snapshot_time(Param.aout[iout],iout,particles,snapshot,mem,0);
	iout++;
      }
      if(iout>=Param.n_aout) break;
#endif //_LIGHTCONE

      // Leap-frog "drift" -- positions updated
      cola_drift(particles,apos1);
    }
  }

#ifdef _LIGHTCONE
  output_lightcone(lightcone,mem);
#endif //_LIGHTCONE

  timer_print();

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

#ifdef _LIGHTCONE
void output_lightcone(Snapshot * const lightcone,Memory mem)
{
  char filebase[256];
  timer_set_category(Snp);
  timer_start(write);

  if(Param.write_snap) {
    sprintf(filebase,"%s",Param.snap_filename);
    write_snapshot(filebase,lightcone);
  }

 //DAM: this will move particles around, which we may not want to do...
  move_particles_snap(lightcone,Param.boxsize,mem.mem1,mem.size1);
  // Density contrast field on PM resolution
  if(Param.write_dens) {
    sprintf(filebase,"%s",Param.dens_filename);
    pm_write_density(filebase,lightcone);
  }
  timer_stop(write);

  msg_printf(normal,"Lightcone written\n");
  timer_set_category(COLA);
}
#endif //_LIGHTCONE

void snapshot_time(const float aout,const int iout, 
		   Particles const * const particles, 
		   Snapshot * const snapshot,
		   Memory mem,int do_lpt)
{
  char filebase[256];      // TODO: make 256 to variable number...?
  timer_set_category(Snp);
  if(do_lpt)
    cola_set_LPT_snapshot(aout,particles,snapshot);
  else
    cola_set_snapshot(aout,particles,snapshot);
  timer_start(write);

  //Do wrap-up and relocation
  move_particles_snap(snapshot,Param.boxsize,mem.mem1,mem.size1);

  // Gadget snapshot for all particles
  // periodic wrapup not done, what about doing move_particle_min here?
  if(Param.write_snap) {
    sprintf(filebase,"%s%03d",Param.snap_filename,iout);
    write_snapshot(filebase,snapshot);
  }

  // Density contrast field on PM resolution
  if(Param.write_dens) {
    sprintf(filebase,"%s%03d",Param.dens_filename,iout);
    pm_write_density(filebase,snapshot);
  }
  timer_stop(write);

  const double z_out=1.0/aout-1.0;
  msg_printf(normal,"snapshot %d written z=%4.2f a=%5.3f\n", 
	     iout+1,z_out,aout);
  timer_set_category(COLA);
}
