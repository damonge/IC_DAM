#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <fftw3-mpi.h>

#include "common.h"

int main(int argc,char* argv[])
{
  MPI_Init(&argc,&argv);
  msg_init();

  //
  // Initialization / Memory allocation
  //						      
  timer_set_category(Init);
  if(argc!=2)
    msg_abort(1,"Error: Parameter file not specified. cola_halo param.init\n");
  read_parameters(argv[1]);

  fftwf_mpi_init();
  comm_init(Param.nc,Param.boxsize);

  cosmo_init(Param.power_spectrum_filename,Param.sigma8,
	     Param.omega_m,1-Param.omega_m,Param.a_final);

  lpt_init(Param.nc);

  MPI_Barrier(MPI_COMM_WORLD);
  int seed=Param.random_seed;

  //
  // LPT
  //
  timer_set_category(LPT);
  lpt_set_displacement(seed,Param.boxsize);

  //
  // Writeout
  //
  timer_set_category(Snp);
  if(Param.nbox_per_side<=0)
    write_snapshot(Param.init_filename,Param.a_final);
  else
    write_snapshot_cola(Param.init_filename,Param.a_final);

  timer_print();
  lpt_end();
  MPI_Finalize();

  return 0;
}
