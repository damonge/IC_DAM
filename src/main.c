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
  cosmo_init();
  lpt_init();
  MPI_Barrier(MPI_COMM_WORLD);

  //
  // LPT
  //
  timer_set_category(LPT);
  lpt_set_displacement();

  //
  // Writeout
  //
  timer_set_category(Snp);
  if(Param.write_as_field)
    write_field();
  else {
    if(Param.nbox_per_side<=0)
      write_snapshot();
    else
      write_snapshot_cola();
  }

  timer_print();
  lpt_end();
  MPI_Finalize();

  return 0;
}
