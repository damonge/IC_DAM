#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

#include "common.h"

static int myrank_;

static int read_parameter_file(char const *fname,Parameters* const param);
static void bcast_string(char* string,int len);

//int read_parameters(const char filename[], Parameters* const param)
int read_parameters(char *filename)
{
  //Only process 0 reads params then communicates
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank_);
  if(myrank_==0) {
    int ret=read_parameter_file(filename,&Param);
    if(ret!=0)
      msg_abort(1001,"Error: Unable to read parameter file: %s\n",filename);
  }

  // Share parameters with other nodes
  MPI_Bcast(&Param,sizeof(Parameters),MPI_BYTE,0,MPI_COMM_WORLD);

  bcast_string(Param.power_spectrum_filename,256);
  bcast_string(Param.init_filename,256);

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
    
    if(!strcmp(s1,"nc="))
      param->nc=atoi(s2);
    else if(!strcmp(s1,"boxsize="))
      param->boxsize=atof(s2);
    else if(!strcmp(s1,"nbox_per_side="))
      param->nbox_per_side=atoi(s2);
    else if(!strcmp(s1,"a_final="))
      param->a_final=atof(s2);
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
    else
      msg_printf("Unknown parameter %s\n",s1);
  }
  fclose(fi);

  return 0;
}

void bcast_string(char* pstring,int len)
{
  const int n=len;

  const int ret2=MPI_Bcast(pstring,n,MPI_CHAR,0,MPI_COMM_WORLD);
  assert(ret2==MPI_SUCCESS);
}
