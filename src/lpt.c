//
// Computes random Gaussian 2LPT displacement on regular particle grid
//
// Based on the code by Roman Scoccimaro, Sebastian Pueblas, Marc Manera et al
// http://cosmo.nyu.edu/roman/2LPT/
//
// link with -lfftw3f_mpi -lfftw3f -lm for single precision FFTW3
//

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <fftw3-mpi.h>
#include <gsl/gsl_rng.h>

#include "common.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971693993751
#endif

static fftwf_plan Inverse_plan[6], Forward_plan;
static fftwf_plan Disp_plan[3], Disp2_plan[3];

static int Nmesh, Nsample;
static int Local_nx, Local_x_start;

static unsigned int *seedtable;

static fftwf_complex *(cdisp[3]), *(cdisp2[3]); // ZA and 2nd order displacements
static float         *(disp[3]), *(disp2[3]);
static fftwf_complex *(cdigrad[6]);
static float         *(digrad[6]);

void lpt_end(void)
{
  for(int axes=0;axes<3;axes++) {
    fftwf_free(cdisp[axes]);
    fftwf_destroy_plan(Disp_plan[axes]);
    fftwf_destroy_plan(Disp2_plan[axes]);
  }
  for(int i=0;i<6;i++) {
    fftwf_free(cdigrad[i]);
    fftwf_destroy_plan(Inverse_plan[i]);
  }
  fftwf_destroy_plan(Forward_plan);
  free(seedtable);
}

// Setup variables for 2LPT initial condition
void lpt_init(void)
{
  const int nc=Param.n_grid;

  // nc: number of mesh per dimension
  ptrdiff_t local_nx,local_x_start;
  ptrdiff_t total_size=fftwf_mpi_local_size_3d(nc,nc,nc,MPI_COMM_WORLD,
					       &local_nx,&local_x_start);
#ifdef _DAM_SAVEMEM
  total_size/=nc;
  total_size*=(nc/2+1);
#endif //_DAM_SAVEMEM

  //
  // Allocate memory
  //

  // allocate memory here
  size_t bytes=0;
  int allocation_failed=0;

  // 1&2 displacement
  for(int axes=0;axes<3;axes++) {
    cdisp[axes]=fftwf_alloc_complex(total_size);
    disp[axes]=(float *)cdisp[axes];
    
    bytes+=sizeof(fftwf_complex)*total_size;
      
    allocation_failed=allocation_failed || (cdisp[axes]==0);
  } 
    
  // 2LPT
  for(int i=0;i<6;i++) {
    cdigrad[i]=(fftwf_complex *)fftwf_alloc_complex(total_size);
    digrad[i]=(float *)cdigrad[i];
    
    bytes+=sizeof(fftwf_complex)*total_size;
    allocation_failed=allocation_failed || (digrad[i]==0);
  } 
    
  if(allocation_failed)
    msg_abort(2003,"Error: Failed to allocate memory for 2LPT."
	      "Tried to allocate %d Mbytes\n",(int)(bytes/(1024*1024)));
  
  msg_printf("%d Mbytes allocated for LPT\n",(int)(bytes/(1024*1024)));

  //
  // FFTW3 plans
  //
  for(int i=0;i<6;++i)
    Inverse_plan[i]=
      fftwf_mpi_plan_dft_c2r_3d(nc,nc,nc,cdigrad[i],digrad[i],
				MPI_COMM_WORLD,FFTW_ESTIMATE);

  Forward_plan=
    fftwf_mpi_plan_dft_r2c_3d(nc,nc,nc,digrad[3],cdigrad[3],
			      MPI_COMM_WORLD,FFTW_ESTIMATE);

  for(int i=0; i<3; ++i) {
    Disp_plan[i]=
      fftwf_mpi_plan_dft_c2r_3d(nc,nc,nc,cdisp[i],disp[i],
				MPI_COMM_WORLD,FFTW_ESTIMATE);
    Disp2_plan[i]=
      fftwf_mpi_plan_dft_c2r_3d(nc,nc,nc,cdisp2[i],disp2[i],
				MPI_COMM_WORLD,FFTW_ESTIMATE);
  }

  Local_nx=Param.local_nx;
  Local_x_start=Param.local_x_start;
  Nmesh=nc;
  Nsample=nc;
  seedtable=malloc(Nmesh*Nmesh*sizeof(unsigned int)); assert(seedtable);
}

void lpt_set_displacement()
{
  msg_printf("Computing LPT displacement fields...\n");
  msg_printf("Random Seed = %d\n",Param.random_seed);

  //
  // Setting constant parameters
  //
  const double boxsize=Param.boxsize;
  const double fac=pow(2*M_PI/boxsize,1.5);

  //
  // Setup random seeds 
  //  complicated way for backward compatibility?
  //
  gsl_rng* random_generator=gsl_rng_alloc(gsl_rng_ranlxd1);
  gsl_rng_set(random_generator,Param.random_seed);

  assert(seedtable);

  for(int i=0;i<Nmesh/2;i++) {//DAM: Why this nonsense??
    for(int j=0;j<i;j++)
      seedtable[i*Nmesh+j]=0x7fffffff*gsl_rng_uniform(random_generator);

    for(int j=0;j<i+1;j++)
      seedtable[j*Nmesh+i]=0x7fffffff*gsl_rng_uniform(random_generator);

    for(int j=0;j<i;j++)
      seedtable[(Nmesh-1-i)*Nmesh+j]=
	0x7fffffff*gsl_rng_uniform(random_generator);

    for(int j=0;j<i+1;j++)
      seedtable[(Nmesh-1-j)*Nmesh+i]=
	0x7fffffff*gsl_rng_uniform(random_generator);

    for(int j=0;j<i;j++)
      seedtable[i*Nmesh+(Nmesh-1-j)]=
	0x7fffffff*gsl_rng_uniform(random_generator);

    for(int j=0;j<i+1;j++)
      seedtable[j*Nmesh+(Nmesh-1-i)]=
	0x7fffffff*gsl_rng_uniform(random_generator);

    for(int j=0;j<i;j++)
      seedtable[(Nmesh-1-i)*Nmesh+(Nmesh-1-j)]=
	0x7fffffff*gsl_rng_uniform(random_generator);

    for(int j=0;j<i+1;j++)
      seedtable[(Nmesh-1-j)*Nmesh+(Nmesh-1-i)]=
	0x7fffffff*gsl_rng_uniform(random_generator);
  }

  // clean the array
  for(int i=0;i<Local_nx;i++)
    for(int j=0;j<Nmesh;j++)
      for(int k=0;k<=Nmesh/2;k++)
	for(int axes=0;axes<3;axes++) {
	  cdisp[axes][(i*Nmesh+j)*(Nmesh/2+1)+k][0]=0.0f;
	  cdisp[axes][(i*Nmesh+j)*(Nmesh/2+1)+k][1]=0.0f;
	}

  double kvec[3];
  double dk=2*M_PI/boxsize;

  for(int i=0;i<Nmesh;i++) {
    int ii=Nmesh-i;
    if(ii==Nmesh)
      ii=0;
    if((i>=Local_x_start && i<(Local_x_start+Local_nx)) ||
       (ii>=Local_x_start && ii<(Local_x_start+Local_nx))) {
      for(int j=0;j<Nmesh;j++) {
	gsl_rng_set(random_generator,seedtable[i*Nmesh+j]);
	
	for(int k=0;k<Nmesh/2;k++) {
	  double phase=gsl_rng_uniform(random_generator)*2*M_PI;
	  double cphase=cos(phase);
	  double sphase=sin(phase);
	  double ampl;
	  do
	    ampl=gsl_rng_uniform(random_generator);
	  while(ampl==0.0);
	  
	  if(i==Nmesh/2 || j==Nmesh/2 || k==Nmesh/2) //DAM: WHY??!!
	    continue;
	  if(i==0 && j==0 && k==0)
	    continue;
	  
	  if(i<Nmesh/2) kvec[0]=i*dk;
	  else kvec[0]=-(Nmesh-i)*dk;
	  
	  if(j<Nmesh/2) kvec[1]=j*dk;
	  else kvec[1]=-(Nmesh-j)*dk;
	  
	  if(k<Nmesh/2) kvec[2]=k*dk;
	  else kvec[2]=-(Nmesh-k)*dk;
	  
	  double kmag2=kvec[0]*kvec[0]+kvec[1]*kvec[1]+kvec[2]*kvec[2];
	  double kmag=sqrt(kmag2);
	  double i_kmag2=1.0/kmag2;
	  
	  if(fabs(kvec[0])*boxsize/(2*M_PI)>Nsample/2) //DAM: WHY??!!
	    continue;
	  if(fabs(kvec[1])*boxsize/(2*M_PI)>Nsample/2)
	    continue;
	  if(fabs(kvec[2])*boxsize/(2*M_PI)>Nsample/2)
	    continue;
		      
	  double p_of_k=PowerSpec(kmag);
		      
	  p_of_k*=-log(ampl);
		      
	  double delta=fac*sqrt(p_of_k);
		      
	  if(k>0) {
	    if(i>=Local_x_start && i<(Local_x_start+Local_nx))
	      for(int axes=0;axes<3;axes++) {
	        cdisp[axes][((i-Local_x_start)*Nmesh+j)*(Nmesh/2+1)+k][0]=
		  -kvec[axes]*i_kmag2*delta*sphase;
		cdisp[axes][((i-Local_x_start)*Nmesh+j)*(Nmesh/2+1)+k][1]=
		  kvec[axes]*i_kmag2*delta*cphase;
	      }
	  }
	  else { // k=0 plane needs special treatment
	    if(i==0) {
	      if(j>=Nmesh/2)
		continue;
	      else {
		if(i>=Local_x_start && i<(Local_x_start+Local_nx)) {
		  int jj=Nmesh-j; // note: j!=0 surely holds at this point
		  
		  for(int axes=0;axes<3;axes++) {
		    cdisp[axes][((i-Local_x_start)*Nmesh+j)*(Nmesh/2+1)+k][0]=
		      -kvec[axes]*i_kmag2*delta*sphase;
		    cdisp[axes][((i-Local_x_start)*Nmesh+j)*(Nmesh/2+1)+k][1]=
		      kvec[axes]*i_kmag2*delta*cphase;
					  
		    cdisp[axes][((i-Local_x_start)*Nmesh+jj)*(Nmesh/2+1)+k][0]=
		      -kvec[axes]*i_kmag2*delta*sphase;
		    cdisp[axes][((i-Local_x_start)*Nmesh+jj)*(Nmesh/2+1)+k][1]=
		      -kvec[axes]*i_kmag2*delta*cphase;
		  }
		}
	      }
	    }
	    else { // here comes i!=0 : conjugate can be on other processor!
	      if(i>=Nmesh/2)
		continue;
	      else {
		ii=Nmesh-i;
		if(ii==Nmesh)
		  ii=0;
		int jj=Nmesh-j;
		if(jj==Nmesh)
		  jj = 0;
		
		if(i>=Local_x_start && i<(Local_x_start+Local_nx))
		  for(int axes=0;axes<3;axes++) {
		    cdisp[axes][((i-Local_x_start)*Nmesh+j)*(Nmesh/2+1)+k][0]=
		      -kvec[axes]*i_kmag2*delta*sphase;
		    cdisp[axes][((i-Local_x_start)*Nmesh+j)*(Nmesh/2+1)+k][1]=
		      kvec[axes]*i_kmag2*delta*cphase;
		  }
		
		if(ii>=Local_x_start && ii<(Local_x_start+Local_nx))
		  for(int axes=0;axes<3;axes++) {
		    cdisp[axes][((ii-Local_x_start)*Nmesh+jj)*(Nmesh/2+1)+k][0]= 
		      -kvec[axes]*i_kmag2*delta*sphase;
		    cdisp[axes][((ii-Local_x_start)*Nmesh+jj)*(Nmesh/2+1)+k][1]= 
		      -kvec[axes]*i_kmag2*delta*cphase;
		  }
	      }
	    }
	  }
	}
      }
    }
  }

  
  //
  // 2nd order LPT
  //
  for(int i=0;i<Local_nx;i++) { 
    if((i+Local_x_start)<Nmesh/2) kvec[0]=(i+Local_x_start)*dk;
    else kvec[0]=-(Nmesh-(i+Local_x_start))*dk;

    for(int j=0;j<Nmesh;j++) {
      if(j<Nmesh/2) kvec[1]=j*dk;
      else kvec[1]=-(Nmesh-j)*dk;
	      
      for(int k=0;k<=Nmesh/2;k++) {
	int coord=(i*Nmesh+j)*(Nmesh/2+1)+k;
	if(k<Nmesh/2) kvec[2]=k*dk;
	else kvec[2]=-(Nmesh-k)*dk;
	      
	// Derivatives of ZA displacement
	// d(dis_i)/d(q_j)  -> sqrt(-1) k_j dis_i
	//DAM:I think the sign is wrong...
	//DAM:But it's ok, since then the source is made up of squares
	cdigrad[0][coord][0]=-cdisp[0][coord][1]*kvec[0]; // disp0,0
	cdigrad[0][coord][1]= cdisp[0][coord][0]*kvec[0];

	cdigrad[1][coord][0]=-cdisp[0][coord][1]*kvec[1]; // disp0,1
	cdigrad[1][coord][1]= cdisp[0][coord][0]*kvec[1];

	cdigrad[2][coord][0]=-cdisp[0][coord][1]*kvec[2]; // disp0,2
	cdigrad[2][coord][1]= cdisp[0][coord][0]*kvec[2];
	      
	cdigrad[3][coord][0]=-cdisp[1][coord][1]*kvec[1]; // disp1,1
	cdigrad[3][coord][1]= cdisp[1][coord][0]*kvec[1];

	cdigrad[4][coord][0]=-cdisp[1][coord][1]*kvec[2]; // disp1,2
	cdigrad[4][coord][1]= cdisp[1][coord][0]*kvec[2];

	cdigrad[5][coord][0]=-cdisp[2][coord][1]*kvec[2]; // disp2,2
	cdigrad[5][coord][1]= cdisp[2][coord][0]*kvec[2];
      }
    }
  }

  msg_printf("Fourier transforming displacement gradient...");

  for(int i=0;i<6;i++)
    fftwf_mpi_execute_dft_c2r(Inverse_plan[i],cdigrad[i],digrad[i]);

  msg_printf("Done.\n");

  // Compute second order source and store it in digrad[3]
  for(int i=0;i<Local_nx;i++) {
    for(int j=0;j<Nmesh;j++) {
      for(int k=0;k<Nmesh;k++) {
	int coord=(i*Nmesh+j)*(2*(Nmesh/2+1))+k;

	digrad[3][coord]=
	  digrad[0][coord]*(digrad[3][coord]+digrad[5][coord])+
	  digrad[3][coord]*digrad[5][coord]-
          digrad[1][coord]*digrad[1][coord]-
          digrad[2][coord]*digrad[2][coord]-
	  digrad[4][coord]*digrad[4][coord];
      }
    }
  }

  msg_printf("Fourier transforming second order source...\n");
  fftwf_mpi_execute_dft_r2c(Forward_plan, digrad[3], cdigrad[3]);

  // The memory allocated for cdigrad[0], [1], and [2] will be used for 
  // 2nd order displacements
  // cdigrad[3] has 2nd order displacement source.

  for(int axes=0;axes<3;axes++) {
    cdisp2[axes]=cdigrad[axes]; 
    disp2[axes]=(float *)cdisp2[axes];
  }

  // Solve Poisson eq. and calculate 2nd order displacements
  for(int i=0;i<Local_nx;i++) {
    if((i+Local_x_start)<Nmesh/2) kvec[0]=(i+Local_x_start)*dk;
    else kvec[0]=-(Nmesh-(i+Local_x_start))*dk;

    for(int j=0;j<Nmesh;j++) {
      if(j<Nmesh/2) kvec[1]=j*dk;
      else kvec[1]=-(Nmesh-j)*dk;
	
      for(int k=0;k<=Nmesh/2;k++) {
	int coord=(i*Nmesh+j)*(Nmesh/2+1)+k;
	
	if(k<Nmesh/2) kvec[2]=k*dk;
	else kvec[2]=-(Nmesh-k)*dk;
	
	double kmag2=kvec[0]*kvec[0]+kvec[1]*kvec[1]+kvec[2]*kvec[2];
	double i_kmag2=1.0/kmag2;
	    
	// cdisp2 = source * k / (sqrt(-1) k^2)
	for(int axes=0;axes<3;axes++) {
	  if(kmag2<=0.0) {
	    cdisp2[axes][coord][0]=0.0;
	    cdisp2[axes][coord][1]=0.0;
	  }
	  else {
	    cdisp2[axes][coord][0]= cdigrad[3][coord][1]*kvec[axes]*i_kmag2;
	    cdisp2[axes][coord][1]=-cdigrad[3][coord][0]*kvec[axes]*i_kmag2;
	  }
	}
      }
    }
  }
      
  // Now, both cdisp, and cdisp2 have the ZA and 2nd order displacements
  for(int axes=0;axes<3;axes++) {  
    msg_printf("Fourier transforming displacements, axis %d.\n",axes);

    fftwf_mpi_execute_dft_c2r(Disp_plan[axes],cdisp[axes],disp[axes]);
    fftwf_mpi_execute_dft_c2r(Disp2_plan[axes],cdisp2[axes],disp2[axes]);
  }

  gsl_rng_free(random_generator);
}

//Writing stuff
static void my_fread(void *ptr,size_t size,size_t count,FILE *stream)
{
  size_t stat=fread(ptr,size,count,stream);
  if(stat!=count)
    msg_abort(123,"Error freading\n");
}

void write_field(void)
{
  char filename[256];
  sprintf(filename,"%s.%d",Param.init_filename,Param.i_node);

  FILE* fp=fopen(filename,"w");
  if(fp==0)
    msg_abort(9000,"Error: Unable to write to file: %s\n",filename);

  const long long npt=((long long)(Nmesh*Nmesh))*Nmesh;
  const int np=Local_nx*Nmesh*Nmesh;
  const double boxsize=Param.boxsize;
  double nmesh3_inv=1.0/pow((double)Nmesh,3.0);

  msg_printf("Writing output to %s.x as field\n",Param.init_filename);

  long long np_send=np,np_total;
  MPI_Reduce(&np_send,&np_total,1,MPI_LONG_LONG, MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Bcast(&np_total,1,MPI_LONG_LONG,0,MPI_COMM_WORLD);

  if(np_total!=npt)
    msg_abort(123,"%ld %ld\n",np_total,npt);

  //Write header
  int blklen=sizeof(double)+sizeof(int);
  fwrite(&blklen,sizeof(blklen),1,fp);
  fwrite(&boxsize,sizeof(double),1,fp);
  fwrite(&Nmesh,sizeof(int),1,fp);
  fwrite(&blklen,sizeof(blklen),1,fp);

  //Writing gridpoints
  float ix[3];
  GridPoint gp;
  blklen=np*sizeof(GridPoint);
  fwrite(&blklen,sizeof(blklen),1,fp);
  for(int i=0;i<Local_nx;i++) {
    ix[0]=i;
    for(int j=0;j<Nmesh;j++) {
      ix[1]=j;
      for(int k=0;k<Nmesh;k++) {
	ix[2]=k;

	for(int axes=0;axes<3;axes++) {
	  float dis=disp[axes][(i*Nmesh+j)*(2*(Nmesh/2+1))+k];
	  float dis2=nmesh3_inv*disp2[axes][(i*Nmesh+j)*(2*(Nmesh/2+1))+k];
	  gp.ix[axes]=ix[axes];
	  gp.dx1[axes]=dis;
	  gp.dx2[axes]=dis2;
	}
	fwrite(&(gp),sizeof(GridPoint),1,fp);
      }
    }
  }
  fwrite(&blklen,sizeof(blklen),1,fp);

  fclose(fp);

  msg_printf("snapshot %s written\n",Param.init_filename);
}

void write_snapshot(void)
{
  float x[3],v[3];
  char filename[256];
  sprintf(filename,"%s.%d",Param.init_filename,Param.i_node);

  FILE* fp=fopen(filename,"w");
  if(fp==0)
    msg_abort(9000,"Error: Unable to write to file: %s\n",filename);

  const long long npt=((long long)(Nmesh*Nmesh))*Nmesh;
  const int np=Local_nx*Nmesh*Nmesh;
  const double boxsize=Param.boxsize;
  const double omega_m=Param.omega_m;
  const double h=Param.h;
  const float dx=Param.boxsize/Nmesh;
  const float vfac=100.0f/Param.a_init;   // km/s; H0= 100 km/s/(h^-1 Mpc)
  const float D1=GrowthFactor(Param.a_init);
  const float D2=GrowthFactor2(Param.a_init);
  const float Dv=Vgrowth(Param.a_init); // dD_{za}/dTau
  const float Dv2=Vgrowth2(Param.a_init); // dD_{2lpt}/dTau
  double nmesh3_inv=1.0/pow((double)Nmesh,3.0);

  msg_printf("Writing output to %s.x with Gadget-1 format\n",Param.init_filename);
#ifdef _LONGIDS
  msg_printf("Longid is used for GADGET snapshot. %d-byte.\n", 
	     sizeof(unsigned long long));
#else //_LONGIDS
  msg_printf("Shortid is used for GADGET snapshot. %d-byte.\n", 
	     sizeof(unsigned int));
#endif //_LONGIDS

  long long np_send=np,np_total;
  MPI_Reduce(&np_send,&np_total,1,MPI_LONG_LONG, MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Bcast(&np_total,1,MPI_LONG_LONG,0,MPI_COMM_WORLD);

  if(np_total!=npt)
    msg_abort(123,"%ld %ld\n",np_total,npt);

  GadgetHeader header; assert(sizeof(GadgetHeader)==256);
  memset(&header,0,sizeof(GadgetHeader));

  const double rho_crit=27.7455;
  const double m=omega_m*rho_crit*pow(boxsize,3.0)/np_total;
  
  header.np[1]=np;
  header.mass[1]=m;
  header.time=Param.a_init;
  header.redshift=1.0/header.time-1;
  header.np_total[1]=(unsigned int)np_total;
  header.np_total_highword[1]=(unsigned int)(np_total >> 32);
  header.num_files=Param.n_nodes;
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
  float x0[3];
  for(int i=0;i<Local_nx;i++) {
    x0[0]=(Local_x_start+i)*dx;
    //    x0[0]=(Local_x_start+i+0.5f)*dx;
    for(int j=0;j<Nmesh;j++) {
      x0[1]=j*dx;
      //      x0[1]=(j+0.5f)*dx;
      for(int k=0;k<Nmesh;k++) {
	x0[2]=k*dx;
	//	x0[2]=(k+0.5f)*dx;

	for(int axes=0;axes<3;axes++) {
	  float dis=disp[axes][(i*Nmesh+j)*(2*(Nmesh/2+1))+k];
	  float dis2=nmesh3_inv*disp2[axes][(i*Nmesh+j)*(2*(Nmesh/2+1))+k];
	  x[axes]=x0[axes]+D1*dis+D2*dis2;
	  if(x[axes]>=boxsize) x[axes]-=boxsize;
	  else if(x[axes]<0) x[axes]+=boxsize;
	}
	fwrite(x,sizeof(float),3,fp);
      }
    }
  }
  fwrite(&blklen,sizeof(blklen),1,fp);

  // velocity
  const float vfac2=1.0/sqrt(Param.a_init);
  fwrite(&blklen,sizeof(blklen),1,fp);
  for(int i=0;i<Local_nx;i++) {
    for(int j=0;j<Nmesh;j++) {
      for(int k=0;k<Nmesh;k++) {
	for(int axes=0;axes<3;axes++) {
	  float dis=disp[axes][(i*Nmesh+j)*(2*(Nmesh/2+1))+k];
	  float dis2=nmesh3_inv*disp2[axes][(i*Nmesh+j)*(2*(Nmesh/2+1))+k];

	  v[axes]=vfac*vfac2*(Dv*dis+Dv2*dis2);
	}
	fwrite(v,sizeof(float),3,fp);
      }
    }
  }
  fwrite(&blklen,sizeof(blklen),1,fp);

  // id
#ifdef _LONGIDS
  blklen=np*sizeof(unsigned long long);
  fwrite(&blklen,sizeof(blklen),1,fp);
  long long id0=(long long)Local_x_start*Nmesh*Nmesh+1;
  for(int i=0;i<np;i++) {
    unsigned long long id_out=id0+i;
    fwrite(&id_out,sizeof(unsigned long long),1,fp); 
  }
  fwrite(&blklen,sizeof(blklen),1,fp);
#else //_LONGIDS
  blklen=np*sizeof(unsigned int);
  fwrite(&blklen,sizeof(blklen),1,fp);
  int id0=Local_x_start*Nmesh*Nmesh+1;
  for(int i=0;i<np;i++) {
    unsigned int id_out=id0+i;
    fwrite(&id_out,sizeof(unsigned int),1,fp); 
  }
  fwrite(&blklen,sizeof(blklen),1,fp);
#endif //_LONGIDS

  fclose(fp);

  msg_printf("snapshot %s written\n",Param.init_filename);
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

void write_snapshot_cola(void)
{
  Particle part;
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

  const long long npt=((long long)(Nmesh*Nmesh))*Nmesh;
  const int np=Local_nx*Nmesh*Nmesh;
  const double boxsize=Param.boxsize;
  const double omega_m=Param.omega_m;
  const double h=Param.h;
  const float dx=Param.boxsize/Nmesh;
  const float vfac=100.0f/Param.a_init;   // km/s; H0= 100 km/s/(h^-1 Mpc)
  const float vfac2=1.0/sqrt(Param.a_init);
  const float D1=GrowthFactor(Param.a_init);
  const float D2=GrowthFactor2(Param.a_init);
  const float Dv=Vgrowth(Param.a_init); // dD_{za}/dTau
  const float Dv2=Vgrowth2(Param.a_init); // dD_{2lpt}/dTau
  double nmesh3_inv=1.0/pow((double)Nmesh,3.0);

  msg_printf("Writing output to %s.x with COLA-boxes format\n",Param.init_filename);
  msg_printf(" There are %d boxes\n",nboxes);
#ifdef _LONGIDS
  msg_printf("Longid is used for GADGET snapshot. %d-byte.\n", 
	     sizeof(unsigned long long));
#else //_LONGIDS
  msg_printf("Shortid is used for GADGET snapshot. %d-byte.\n", 
	     sizeof(unsigned int));
#endif //_LONGIDS

  long long np_send=np,np_total;
  MPI_Reduce(&np_send,&np_total,1,MPI_LONG_LONG, MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Bcast(&np_total,1,MPI_LONG_LONG,0,MPI_COMM_WORLD);

  if(np_total!=npt)
    msg_abort(123,"%ld %ld\n",np_total,npt);

  GadgetHeader header; assert(sizeof(GadgetHeader)==256);
  memset(&header,0,sizeof(GadgetHeader));

  const double rho_crit=27.7455;
  const double m=omega_m*rho_crit*pow(boxsize,3.0)/np_total;
  
  header.mass[1]=m;
  header.time=Param.a_init;
  header.redshift=1.0/header.time-1;
  header.boxsize=boxsize/nbside;
  header.omega0=omega_m;
  header.omega_lambda=1.0-omega_m;
  header.hubble_param=h;
  header.flag_gadgetformat=0;

  float x0[3];
#ifdef _LONGIDS
  long long id=(long long)Local_x_start*Nmesh*Nmesh+1;
#else //_LONGIDS
  int id=Local_x_start*Nmesh*Nmesh+1;
#endif //_LONGIDS
  int thisnode=Param.i_node;
  double inv_l_subbox=nbside/boxsize;
  double l_subbox=boxsize/nbside;
  for(int i=0;i<Local_nx;i++) {
    x0[0]=(Local_x_start+i)*dx;
    //    x0[0]=(Local_x_start+i+0.5f)*dx;
    for(int j=0;j<Nmesh;j++) {
      x0[1]=j*dx;
      //      x0[1]=(j+0.5f)*dx;
      for(int k=0;k<Nmesh;k++) {
	x0[2]=k*dx;
	//	x0[2]=(k+0.5f)*dx;

	int index_box;
	for(int axes=0;axes<3;axes++) {
	  float dis=disp[axes][(i*Nmesh+j)*(2*(Nmesh/2+1))+k];
	  float dis2=nmesh3_inv*disp2[axes][(i*Nmesh+j)*(2*(Nmesh/2+1))+k];

	  //Compute position and velocity
	  part.x[axes]=x0[axes]+D1*dis+D2*dis2;
	  part.v[axes]=vfac*vfac2*(Dv*dis+Dv2*dis2);
	  part.dx1[axes]=dis;
	  part.dx2[axes]=dis2;

	  if(part.x[axes]<0) part.x[axes]+=boxsize;
	  if(part.x[axes]>=boxsize) part.x[axes]-=boxsize;
	  if(part.x[axes]<0) part.x[axes]+=boxsize;

	  //Compute which subbox
	  int ib=(int)(inv_l_subbox*part.x[axes]);
	  if(ib<0 || ib>=nbside)
	    msg_abort(123,"Wrong subbox index! %d %d\n",ib,nbside);
	  ibox[axes]=ib;

	  //Compute new origin
	  part.x[axes]-=ibox[axes]*l_subbox;
	}
	part.id=id;
	id++;
	index_box=ibox[0]+nbside*(ibox[1]+nbside*ibox[2]);

	//Open file if not created yet
	if(box_touched[index_box]==0) {
	  sprintf(fname,"%s_box%dp%dp%d.%d",Param.init_filename,
		  ibox[0],ibox[1],ibox[2],thisnode);
	  file_array[index_box]=new_gadget_file(fname,header);
	  box_touched[index_box]=1;
	}


	np_in_box[index_box]++;
	fwrite(&part,sizeof(Particle),1,file_array[index_box]);
      }
    }
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
      msg_printf("%llu particles in box (%d,%d,%d), from %d nodes\n",
		 np_in_box_total[i],ibox[0],ibox[1],ibox[2],box_touched_total[i]);
      ntot+=np_in_box_total[i];
    }
    msg_printf("Total: %llu particles\n",ntot);
  }

  //Fix headers and particle numbers
  for(int i=0;i<nboxes;i++) {
    if(np_in_box[i]>0) {
      //Compute box coordinates
      ibox[0]=i%nbside;
      ibox[1]=((i-ibox[0])/nbside)%nbside;
      ibox[2]=(i-ibox[0]-ibox[1]*nbside)/(nbside*nbside);

      //Open file
      sprintf(fname,"%s_box%dp%dp%d.%d",Param.init_filename,
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
  msg_printf("snapshot %s written\n",Param.init_filename);
}
