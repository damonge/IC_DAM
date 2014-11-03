#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <fftw3-mpi.h>

#include "common.h"

static int myrank_;

static int read_parameter_file(char const *fname,Parameters* const param);
static void bcast_string(char* string,int len);

void set_default_params()
{
  Param.n_grid=128;
  Param.boxsize=125.;
  Param.nbox_per_side=0;
  Param.a_init=1.0;
  Param.random_seed=1234;
  Param.omega_m=0.315;
  Param.sigma8=0.83;
  Param.h=0.67;
  sprintf(Param.power_spectrum_filename,"default");
  sprintf(Param.init_filename,"default");
  Param.scale_cutoff=-1.0;
  Param.cut_ls=0;
  Param.write_as_field=0;
}

//int read_parameters(const char filename[], Parameters* const param)
int read_parameters(char *filename)
{
  //Only process 0 reads params then communicates
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank_);
  if(myrank_==0) {
    set_default_params();
    read_parameter_file(filename,&Param);
  }

  // Share parameters with other nodes
  MPI_Bcast(&Param,sizeof(Parameters),MPI_BYTE,0,MPI_COMM_WORLD);

  bcast_string(Param.power_spectrum_filename,256);
  bcast_string(Param.init_filename,256);

  //Get MPI parameters
  MPI_Comm_rank(MPI_COMM_WORLD,&(Param.i_node));
  MPI_Comm_size(MPI_COMM_WORLD,&(Param.n_nodes));
  Param.i_node_left=(Param.i_node-1+Param.n_nodes)%Param.n_nodes;
  Param.i_node_right=(Param.i_node+1)%Param.n_nodes;

  //Get FFTW parameters
  fftwf_mpi_local_size_3d(Param.n_grid,Param.n_grid,Param.n_grid,
			  MPI_COMM_WORLD,&(Param.local_nx),
			  &(Param.local_x_start));

#ifdef _DEBUG
  int *local_x_table=malloc(Param.n_nodes*2*sizeof(int));
  int *local_nx_table=local_x_table+Param.n_nodes;
  
  MPI_Allgather(&(Param.local_nx),1,MPI_INT,local_nx_table,
		1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&(Param.local_x_start),1,MPI_INT,local_x_table,
		1,MPI_INT,MPI_COMM_WORLD);

  for(int i=0;i<Param.n_nodes;i++)
    msg_printf("LPT Task=%d x=%d..%d\n",i,local_x_table[i],
	       local_x_table[i]+local_nx_table[i]-1);
  free(local_x_table);
#endif //_DEBUG

  return 0;
}

static int linecount(FILE *f)
{
  //////
  // Counts #lines from file
  int i0=0;
  char ch[1000];
  while((fgets(ch,sizeof(ch),f))!=NULL) {
    i0++;
  }
  return i0;
}

static int read_parameter_file(char const *fname,Parameters *param)
{
  FILE *fi;
  int n_lin,ii;
  
  //Read parameters from file
  fi=fopen(fname,"r");
  if(fi==NULL)
    msg_abort(1000, "Error: couldn't open file %s\n",fname);

  // Fix to 1 in case it is not present
  param->nbox_per_side=1;

  n_lin=linecount(fi);
  rewind(fi);
  for(ii=0;ii<n_lin;ii++) {
    char s0[512],s1[64],s2[256];
    if(fgets(s0,sizeof(s0),fi)==NULL)
      msg_abort(1001,"Error reading line %d, file %s\n",ii+1,fname);
    if((s0[0]=='#')||(s0[0]=='\n')) continue;
    int sr=sscanf(s0,"%s %s",s1,s2);
    if(sr!=2)
      msg_abort(1002,"Error reading line %d, file %s\n",ii+1,fname);
    
    if(!strcmp(s1,"n_grid="))
      param->n_grid=atoi(s2);
    else if(!strcmp(s1,"boxsize="))
      param->boxsize=atof(s2);
    else if(!strcmp(s1,"nbox_per_side="))
      param->nbox_per_side=atoi(s2);
    else if(!strcmp(s1,"a_init="))
      param->a_init=atof(s2);
    else if(!strcmp(s1,"random_seed="))
      param->random_seed=atoi(s2);
    else if(!strcmp(s1,"omega_m="))
      param->omega_m=atof(s2);
    else if(!strcmp(s1,"h="))
      param->h=atof(s2);
    else if(!strcmp(s1,"sigma8="))
      param->sigma8=atof(s2);
    else if(!strcmp(s1,"np_alloc_factor="))
      param->np_alloc_factor=atof(s2);
    else if(!strcmp(s1,"powerspectrum="))
      sprintf(param->power_spectrum_filename,"%s",s2);
    else if(!strcmp(s1,"init_fname="))
      sprintf(param->init_filename,"%s",s2);
    else if(!strcmp(s1,"scale_cutoff="))
      param->scale_cutoff=atof(s2);
    else if(!strcmp(s1,"cut_ls="))
      param->cut_ls=atoi(s2);
    else if(!strcmp(s1,"write_as_field="))
      param->write_as_field=atoi(s2);
    else
      msg_printf("Unknown parameter %s\n",s1);
  }
  fclose(fi);

  //Sanity checks
  if((param->n_grid<=0) ||
     (param->boxsize<=0) ||
     (param->nbox_per_side<0) ||
     (param->a_init<=0) ||
     (param->omega_m<=0) ||
     (param->sigma8<=0) ||
     (param->h<=0) ||
     (param->np_alloc_factor<=1.0)) {
    msg_abort(123,"One or more parameters have nonsensical values\n");
  }

  return 0;
}

void bcast_string(char* pstring,int len)
{
  const int n=len;

  const int ret2=MPI_Bcast(pstring,n,MPI_CHAR,0,MPI_COMM_WORLD);
  assert(ret2==MPI_SUCCESS);
}
