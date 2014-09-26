#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "common.h"

void write_snapshot(const char filebase[],Snapshot const * const snapshot)
{
  char filename[256];
  sprintf(filename,"%s.%d",filebase,comm_this_node());

  FILE* fp=fopen(filename,"w");
  if(fp==0)
    msg_abort(9000,"Error: Unable to write to file: %s\n",filename);

  ParticleMinimum* const p=snapshot->p;
  const int np=snapshot->np_local;
  const double boxsize=Param.boxsize;
  const double omega_m=Param.omega_m;
  const double h=Param.h;

#ifdef _LONGIDS
  msg_printf(normal,"Longid is used for GADGET snapshot. %d-byte.\n", 
	     sizeof(unsigned long long));
#else //_LONGIDS
  msg_printf(normal,"ID is %d-byte unsigned int\n",sizeof(unsigned int));
#endif //_LONGIDS

  long long np_send=np,np_total;
  MPI_Reduce(&np_send,&np_total,1,MPI_LONG_LONG, MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Bcast(&np_total,1,MPI_LONG_LONG,0,MPI_COMM_WORLD);

  if(np_total!=snapshot->np_total)
    msg_abort(123,"%ld %ld\n",np_total,snapshot->np_total);

  GadgetHeader header; assert(sizeof(GadgetHeader)==256);
  memset(&header,0,sizeof(GadgetHeader));

  const double rho_crit=27.7455;
  const double m=omega_m*rho_crit*pow(boxsize,3.0)/np_total;
  
  header.np[1]=np;
  header.mass[1]=m;
  header.time=snapshot->a;
  header.redshift=1.0/header.time-1;
  header.np_total[1]=(unsigned int)np_total;
  header.np_total_highword[1]=(unsigned int)(np_total >> 32);
  header.num_files=comm_nnode();
  header.boxsize=boxsize;
  header.omega0=omega_m;
  header.omega_lambda=1.0-omega_m;
  header.hubble_param=h;


  int blklen=sizeof(GadgetHeader);
  fwrite(&blklen,sizeof(blklen),1,fp);
  fwrite(&header,sizeof(GadgetHeader),1,fp);
  fwrite(&blklen,sizeof(blklen),1,fp);

  // position
  blklen=np*sizeof(float)*3;
  fwrite(&blklen,sizeof(blklen),1,fp);
  for(int i=0;i<np;i++) {
    fwrite(p[i].x,sizeof(float),3,fp);
  }
  fwrite(&blklen,sizeof(blklen),1,fp);

  // velocity
  const float vfac=1.0/sqrt(snapshot->a); // Gadget convention

  fwrite(&blklen,sizeof(blklen),1,fp);
  for(int i=0;i<np;i++) {
    float vout[]={vfac*p[i].v[0],vfac*p[i].v[1],vfac*p[i].v[2]};
    fwrite(vout,sizeof(float),3,fp);
  }
  fwrite(&blklen,sizeof(blklen),1,fp);

  // id
#ifdef _LONGIDS
  blklen=np*sizeof(unsigned long long);
  fwrite(&blklen,sizeof(blklen),1,fp);
  for(int i=0;i<np;i++) {
    unsigned long long id_out=p[i].id;
    fwrite(&id_out,sizeof(unsigned long long),1,fp); 
  }
#else //_LONGIDS
  blklen=np*sizeof(unsigned int);
  fwrite(&blklen,sizeof(blklen),1,fp);
  for(int i=0;i<np;i++) {
    unsigned int id_out=p[i].id;
    fwrite(&id_out,sizeof(unsigned int),1,fp); 
  }
#endif //_LONGIDS
  fwrite(&blklen,sizeof(blklen),1,fp);

  // displacement1
  blklen=np*sizeof(float)*3;
  fwrite(&blklen,sizeof(blklen),1,fp);
  for(int i=0;i<np;i++) {
    fwrite(p[i].dx1,sizeof(float),3,fp);
  }
  fwrite(&blklen,sizeof(blklen),1,fp);

  // displacement2
  blklen=np*sizeof(float)*3;
  fwrite(&blklen,sizeof(blklen),1,fp);
  for(int i=0;i<np;i++) {
    fwrite(p[i].dx2,sizeof(float),3,fp);
  }
  fwrite(&blklen,sizeof(blklen),1,fp);

  fclose(fp);

  msg_printf(normal, "snapshot %s written\n", filebase);
}
