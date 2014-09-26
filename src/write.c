#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "common.h"

static void my_fread(void *ptr,size_t size,size_t count,FILE *stream)
{
  size_t stat=fread(ptr,size,count,stream);
  if(stat!=count)
    msg_abort(123,"Error freading\n");
}

void write_snapshot(const char filebase[],Snapshot const * const snapshot)
{
  char filename[256];
  sprintf(filename,"%s.%d",filebase,comm_this_node());

  FILE* fp=fopen(filename,"w");
  if(fp==0)
    msg_abort(9000,"Error: Unable to write to file: %s\n",filename);

  Particle* const p=snapshot->p;
  const int np=snapshot->np_local;
  const double boxsize=Param.boxsize;
  const double omega_m=Param.omega_m;
  const double h=Param.h;

  msg_printf(normal,"Longid is used for GADGET snapshot. %d-byte.\n", 
	     sizeof(unsigned long long));

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
  header.flag_gadgetformat=1;

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
  blklen=np*sizeof(unsigned long long);
  fwrite(&blklen,sizeof(blklen),1,fp);
  for(int i=0;i<np;i++) {
    unsigned long long id_out=p[i].id;
    fwrite(&id_out,sizeof(unsigned long long),1,fp); 
  }
  fwrite(&blklen,sizeof(blklen),1,fp);

  fclose(fp);

  msg_printf(normal, "snapshot %s written\n", filebase);
}

static FILE *new_gadget_file(char *fname,GadgetHeader header)
{
  FILE *f=fopen(fname,"w");
  int blklen=sizeof(GadgetHeader);
  fwrite(&blklen,sizeof(blklen),1,f);
  fwrite(&header,sizeof(GadgetHeader),1,f);
  fwrite(&blklen,sizeof(blklen),1,f);

  //Beginning of first block
  blklen=0;
  fwrite(&blklen,sizeof(blklen),1,f);

  return f;
}

static void end_gadget_file(FILE *f)
{
  int blklen=0;
  fwrite(&blklen,sizeof(blklen),1,f);
  fclose(f);
}

void write_snapshot_cola(const char filebase[],Snapshot const * const snapshot)
{
  char fname[256];
  int blklen,ibox[3];
  unsigned long long *np_in_box,*np_in_box_total;
  int *box_touched,*box_touched_total;
  FILE **file_array;
  int nbside=Param.nbox_per_side;
  int nboxes=nbside*nbside*nbside;

  np_in_box=(unsigned long long *)calloc(nboxes,sizeof(unsigned long long));
  np_in_box_total=(unsigned long long *)calloc(nboxes,sizeof(unsigned long long));
  box_touched=(int *)calloc(nboxes,sizeof(int));
  box_touched_total=(int *)calloc(nboxes,sizeof(int));
  file_array=(FILE **)malloc(nboxes*sizeof(FILE *));

  Particle* const p=snapshot->p;
  const int np=snapshot->np_local;
  const double boxsize=Param.boxsize;
  const double omega_m=Param.omega_m;
  const double h=Param.h;

  msg_printf(normal,"Longid is used for GADGET snapshot. %d-byte.\n", 
	     sizeof(unsigned long long));

  long long np_send=np,np_total;
  MPI_Reduce(&np_send,&np_total,1,MPI_LONG_LONG, MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Bcast(&np_total,1,MPI_LONG_LONG,0,MPI_COMM_WORLD);

  if(np_total!=snapshot->np_total)
    msg_abort(123,"%ld %ld\n",np_total,snapshot->np_total);

  GadgetHeader header; assert(sizeof(GadgetHeader)==256);
  memset(&header,0,sizeof(GadgetHeader));

  const double rho_crit=27.7455;
  const double m=omega_m*rho_crit*pow(boxsize,3.0)/np_total;
  
  header.mass[1]=m;
  header.time=snapshot->a;
  header.redshift=1.0/header.time-1;
  header.boxsize=boxsize;
  header.omega0=omega_m;
  header.omega_lambda=1.0-omega_m;
  header.hubble_param=h;
  header.flag_gadgetformat=0;

  int thisnode=comm_this_node();
  double inv_l_subbox=nbside/boxsize;
  double l_subbox=boxsize/nbside;
  for(int i=0;i<np;i++) {
    int index_box;
    for(int axes=0;axes<3;axes++) {
      // Wrap in box
      if(p[i].x[axes]>=boxsize) p[i].x[axes]-=boxsize;
      else if(p[i].x[axes]<0) p[i].x[axes]+=boxsize;

      //Compute which box
      ibox[axes]=(int)(inv_l_subbox*p[i].x[axes]);

      //Compute new origin
      p[i].x[axes]-=ibox[axes]*l_subbox;
    }
    index_box=ibox[0]+nbside*(ibox[1]+nbside*ibox[2]);
    
    if(box_touched[index_box]==0) {
      sprintf(fname,"%s_box%dp%dp%d.%d",filebase,
	      ibox[0],ibox[1],ibox[2],thisnode);
      file_array[index_box]=new_gadget_file(fname,header);
      box_touched[index_box]=1;
    }

    np_in_box[index_box]++;
    fwrite(&(p[i]),sizeof(Particle),1,file_array[index_box]);
  }

  for(int i=0;i<nboxes;i++) {
    if(box_touched[i]>0)
      end_gadget_file(file_array[i]);
  }

  MPI_Allreduce(np_in_box,np_in_box_total,nboxes,
		MPI_UNSIGNED_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(box_touched,box_touched_total,nboxes,
		MPI_INT,MPI_SUM,MPI_COMM_WORLD);

  if(thisnode==0) {
    unsigned long long ntot=0;
    for(int i=0;i<nboxes;i++) {
      ibox[0]=i%nbside;
      ibox[1]=((i-ibox[0])/nbside)%nbside;
      ibox[2]=(i-ibox[0]-ibox[1]*nbside)/(nbside*nbside);
      msg_printf(normal,"%llu particles in box (%d,%d,%d), from %d nodes\n",
		 np_in_box_total[i],ibox[0],ibox[1],ibox[2],box_touched_total[i]);
      ntot+=np_in_box_total[i];
    }
    msg_printf(normal,"Total: %llu particles\n",ntot);
  }

  //Fix headers and particle numbers
  for(int i=0;i<nboxes;i++) {
    if(np_in_box[i]>0) {
      //Compute box coordinates
      ibox[0]=i%nbside;
      ibox[1]=((i-ibox[0])/nbside)%nbside;
      ibox[2]=(i-ibox[0]-ibox[1]*nbside)/(nbside*nbside);

      //Open file
      sprintf(fname,"%s_box%dp%dp%d.%d",filebase,
	      ibox[0],ibox[1],ibox[2],thisnode);
      file_array[i]=fopen(fname,"r+");

      //Read header
      my_fread(&blklen,sizeof(int),1,file_array[i]);
      if(blklen!=sizeof(GadgetHeader))
	msg_abort(123,"shit1!\n");
      my_fread(&header,sizeof(GadgetHeader),1,file_array[i]);
      my_fread(&blklen,sizeof(int),1,file_array[i]);
      if(blklen!=sizeof(GadgetHeader))
	msg_abort(123,"shit2!\n");
      rewind(file_array[i]);

      header.np[1]=(int)(np_in_box[i]);
      header.np_total[1]=(unsigned int)(np_in_box_total[i]);
      header.np_total_highword[1]=(unsigned int)(np_in_box_total[i] >> 32);
      header.num_files=box_touched_total[i];
      my_fread(&blklen,sizeof(int),1,file_array[i]);
      if(blklen!=sizeof(GadgetHeader))
	msg_abort(123,"shit3! %d %d\n",blklen,sizeof(GadgetHeader));
      fwrite(&header,sizeof(GadgetHeader),1,file_array[i]);
      my_fread(&blklen,sizeof(int),1,file_array[i]);
      if(blklen!=sizeof(GadgetHeader))
	msg_abort(123,"shit4! %d %d\n",blklen,sizeof(GadgetHeader));

      long int block_size=np_in_box[i]*sizeof(Particle);
      blklen=np_in_box[i]*sizeof(Particle);
      fwrite(&blklen,sizeof(int),1,file_array[i]);
      fseek(file_array[i],block_size,SEEK_CUR);
      fwrite(&blklen,sizeof(int),1,file_array[i]);
      fclose(file_array[i]);
    }
  }

  free(box_touched);
  free(box_touched_total);
  free(np_in_box);
  free(np_in_box_total);
  free(file_array);
  msg_printf(normal, "snapshot %s written\n", filebase);
}
