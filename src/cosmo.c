//
// Reads CAMB matter power spectrum camb_matterpower.dat, and
// provides power spectrum to 2LPT calculation in lpt.c
//
// Based on N-GenIC power.c by Volker Springel
//   http://www.mpa-garching.mpg.de/gadget/right.html#ICcode
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include "common.h"

#define WORKSIZE 100000

static double Norm;
static int NPowerTable;
static double r_tophat;
static double Omega, OmegaLambda;

static struct pow_table
{
  double logk, logD;
} *PowerTable;

#define N_A_GROWTH 1000
static struct growth_table
{
  double a,gf;
} *GrowthTable;

#ifdef _LIGHTCONE
#define N_RCOM 100
#define AMIN 0.005 //ZMAX=199
double R2max;
double Inv_dr2;
float ObsPos[3];

static struct rcom_table
{
  double r2,a;
} *RcomTable;

float x2a(float const * const pos)
{
  float x=pos[0]-ObsPos[0];
  float y=pos[1]-ObsPos[1];
  float z=pos[2]-ObsPos[2];
  float r2=x*x+y*y+z*z;

  if(r2<0||r2>R2max)
    msg_abort(3001,"Wrong radius %lf\n",sqrt(r2));
  if(r2==0) return 1;
  else if(r2==R2max) return AMIN;
  else {
    int ibin=(int)(r2*Inv_dr2);
    double u=(r2-RcomTable[ibin].r2)*Inv_dr2;
    double aout=(1-u)*RcomTable[ibin].a+u*RcomTable[ibin+1].a;

    return (float)aout;
  }
}

static double integrand_rcom(double a,void *params)
{
  return 1/sqrt(a*(Omega+OmegaLambda*a*a*a+(1-Omega-OmegaLambda)*a));
}

static double rcom_of_a(double a)
{
  double integral,errintegral;
  gsl_function integrand;
  size_t sdum;
  integrand.function=&integrand_rcom;
  integrand.params=NULL;

  gsl_integration_qng(&integrand,a,1,0,1E-6,
		      &integral,&errintegral,&sdum);
  return 2997.92458*integral;
}

#define RTOL 1.0E-5
static double find_a(double r2)
{
  if(r2==0) return 1;
  else {
    double am;
    double ai=AMIN;
    double af=1.0;
    double err_r=1.0;
    double r_in=sqrt(r2);

    while(err_r>RTOL) {
      double rm;
      am=0.5*(ai+af);
      rm=rcom_of_a(am);

      if(rm<r_in) af=am;
      else ai=am;
      
      err_r=fabs(rm/r_in-1);
    }

    return am;
  }
}

static void set_lightcone(void)
{
  int ii;

  for(ii=0;ii<3;ii++) ObsPos[ii]=Param.pos_obs[ii];

  R2max=pow(rcom_of_a(AMIN),2);
  Inv_dr2=N_RCOM/R2max;
  RcomTable=malloc(N_RCOM*sizeof(struct rcom_table));

  for(ii=0;ii<N_RCOM;ii++) {
    double r2=ii/Inv_dr2;
    double a=find_a(r2);

    RcomTable[ii].r2=r2;
    RcomTable[ii].a=a;
  }
}
#endif //_LIGHTCONE

static void read_power_table_camb(const char filename[])
{
  double k, p;
  double fac= 1.0/(2.0*M_PI*M_PI);
  Norm= 1.0;

  FILE* fp= fopen(filename,"r");
  if(!fp)
    msg_abort(3000, "Error: unable to read input power spectrum: %s",
	      filename);

  NPowerTable = 0;
  do {
    if(fscanf(fp,"%lg %lg ",&k,&p)==2)
      NPowerTable++;
    else
      break;
  } while(1);

  msg_printf(verbose,
	     "Found %d pairs of values in input spectrum table\n",NPowerTable);

  PowerTable=malloc(NPowerTable*sizeof(struct pow_table));

  rewind(fp);

  int n=0;
  do {
    if(fscanf(fp," %lg %lg ",&k,&p)==2) {
      PowerTable[n].logk=log10(k);
      PowerTable[n].logD=log10(fac*k*k*k*p);
      n++;
    }
    else
      break;
  } while(1);
  assert(NPowerTable==n);

  fclose(fp);
}

double PowerSpec(const double k)
{
  //DAM: Returns  P_k/(2*pi)^3, where P_k is CAMB's power spectrum
  const double logk=log10(k);

  if(logk<PowerTable[0].logk || logk>PowerTable[NPowerTable-1].logk) //DAM: this could be extrapolated
    return 0;

  int binlow=0;
  int binhigh=NPowerTable-1;

  while(binhigh-binlow>1) {
    int binmid=(binhigh+binlow) / 2;
    if(logk<PowerTable[binmid].logk)
      binhigh=binmid;
    else
      binlow=binmid;
  }

  const double dlogk=PowerTable[binhigh].logk-PowerTable[binlow].logk;
  assert(dlogk>0.0);

  const double u=(logk-PowerTable[binlow].logk)/dlogk;

  const double logD=(1-u)*PowerTable[binlow].logD+u*PowerTable[binhigh].logD;

  const double Delta2=pow(10.0,logD);

  double P=Norm*Delta2/(4.0*M_PI*k*k*k);

  return P;
}

static double sigma2_int(double k, void *param)
{
  double kr,kr3,kr2,w,x;

  kr=r_tophat*k;
  kr2=kr*kr;
  kr3=kr2*kr;

  if(kr<1e-8)
    return 0;

  w=3*(sin(kr)/kr3-cos(kr)/kr2);
  x=4*M_PI*k*k*w*w*PowerSpec(k);

  return x;
}

static double TopHatSigma2(double R)
{
  double result,abserr;
  gsl_integration_workspace *workspace;
  gsl_function F;

  workspace=gsl_integration_workspace_alloc(WORKSIZE);

  F.function=&sigma2_int;

  r_tophat=R;

  gsl_integration_qag(&F,0,500.0*1/R,0,1.0e-4,WORKSIZE,
		      GSL_INTEG_GAUSS41,workspace,&result,
		      &abserr);
  // high precision caused error
  gsl_integration_workspace_free(workspace);

  return result;

  // note: 500/R is here chosen as (effectively) infinity integration boundary
}

static double normalize_power(const double sigma8)
{
  //DAM: sigma8 is not being taken into account!
  // Assume that input power spectrum already has a proper sigma8
  const double R8=8.0; // 8 Mpc

  double res=TopHatSigma2(R8); 
  double sigma8_input=sqrt(res);

  if(abs(sigma8_input-sigma8)/sigma8>0.05)
    msg_abort(3010,"Input sigma8 %f is far from target sigma8 %f\n",
	      sigma8_input,sigma8);

  msg_printf(info,"Input power spectrum sigma8 %f\n",sigma8_input);

  return 1.0;
}

static double omega_a(const double a)
{
  return Omega/(Omega+(1-Omega-OmegaLambda)*a+
		 OmegaLambda*a*a*a);
}

double Qfactor(const double a)
{
  double acube=a*a*a;
  return sqrt(Omega*acube+OmegaLambda*acube*acube+
	      (1-Omega-OmegaLambda)*acube*a);
}

double GrowthFactor(const double a)
{
  if(a<0||a>1)
    msg_abort(3000,"Wrong scale factor %lf\n",a);
  if(a==1) return 1;
  else if(a==0) return 0;
  else {
    int ibin=(int)(a*N_A_GROWTH);
    double u=(a-GrowthTable[ibin].a)*N_A_GROWTH;
    double gfac=(1-u)*GrowthTable[ibin].gf+u*GrowthTable[ibin+1].gf;

    return gfac;
  }
}

double GrowthFactor2(const double a)
{
  double gf1=GrowthFactor(a);
  double om=omega_a(a);

  return -0.42857142857*gf1*gf1*pow(om,-0.00699300699); //-3/7*D_1^2*Omega_M^{-1/143}
}

double Vgrowth(const double a) //Returns dD1/dTau
{
  double om=omega_a(a);
  double q=Qfactor(a);
  double f1=pow(om,0.555);
  double d1=GrowthFactor(a);
  
  return q*d1*f1/a;
}

double Vgrowth2(const double a) //Returns dD2/dTau
{
  double om=omega_a(a);
  double q=Qfactor(a);
  double f2=2*pow(om,0.545454);
  double d2=GrowthFactor2(a);
  
  return q*d2*f2/a;
}

static double growth_int(double a,void *param)
{
  return pow(a/(Omega+(1-Omega-OmegaLambda)*a+OmegaLambda*a*a*a),1.5);
}

static double growth(double a)
{
  double hubble_a;

  hubble_a=sqrt(Omega/(a*a*a)+(1-Omega-OmegaLambda)/(a*a)+OmegaLambda);

  double result, abserr;
  gsl_integration_workspace *workspace;
  gsl_function F;

  workspace=gsl_integration_workspace_alloc(WORKSIZE);

  F.function=&growth_int;

  gsl_integration_qag(&F,0,a,0,1.0e-8,WORKSIZE,GSL_INTEG_GAUSS41, 
		      workspace,&result,&abserr);

  gsl_integration_workspace_free(workspace);

  return hubble_a*result;
}

static void get_growth_factor(void)
{
  int ii;
  double g0=growth(1.0);
  GrowthTable=malloc(N_A_GROWTH*sizeof(struct growth_table));

  for(ii=0;ii<N_A_GROWTH;ii++) {
    double a=(double)ii/N_A_GROWTH;
    double gf=growth(a)/g0;

    GrowthTable[ii].a=a;
    GrowthTable[ii].gf=gf;
  }
}

void cosmo_init(const char filename[],const double sigma8,
		const double omega_m,const double omega_lambda)
{
  Omega=omega_m;
  OmegaLambda=omega_lambda;

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  if(myrank==0) {
    read_power_table_camb(filename);
    Norm=normalize_power(sigma8);
    get_growth_factor();
#ifdef _LIGHTCONE
    set_lightcone();
#endif //_LIGHTCONE
  }

  msg_printf(normal,"Powerspecectrum file: %s\n",filename);

  MPI_Bcast(&NPowerTable,1,MPI_INT,0,MPI_COMM_WORLD);
  if(myrank!=0) {
    PowerTable=malloc(NPowerTable*sizeof(struct pow_table));
    GrowthTable=malloc(N_A_GROWTH*sizeof(struct growth_table));
#ifdef _LIGHTCONE
    RcomTable=malloc(N_RCOM*sizeof(struct rcom_table));
#endif // _LIGHTCONE
  }

  MPI_Bcast(PowerTable,NPowerTable*sizeof(struct pow_table),MPI_BYTE,0,
	    MPI_COMM_WORLD);
  MPI_Bcast(&Norm,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Bcast(GrowthTable,N_A_GROWTH*sizeof(struct growth_table),MPI_BYTE,0,
	    MPI_COMM_WORLD);
#ifdef _LIGHTCONE
  MPI_Bcast(RcomTable,N_RCOM*sizeof(struct rcom_table),MPI_BYTE,0,
	    MPI_COMM_WORLD);
  MPI_Bcast(&R2max,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Bcast(&Inv_dr2,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Bcast(ObsPos,3,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif // _LIGHTCONE
}

void set_a_final(void)
{
#ifdef _LIGHTCONE
  int axes;
  double a0,af;
  float r2max=0,r2min=0;

  for(axes=0;axes<3;axes++) {
    float dist_max,dist_min;

    if(Param.pos_obs[axes]<0) dist_min=-Param.pos_obs[axes];
    else if(Param.pos_obs[axes]>=Param.boxsize) dist_min=Param.pos_obs[axes]-Param.boxsize;
    else dist_min=0;

    if(Param.pos_obs[axes]<0) dist_max=Param.boxsize-Param.pos_obs[axes];
    else if(Param.pos_obs[axes]>0.5*Param.boxsize) dist_max=Param.pos_obs[axes];
    else dist_max=Param.boxsize-Param.pos_obs[axes];

    r2max+=dist_max*dist_max;
    r2min+=dist_min*dist_min;
  }
  a0=find_a(r2max);
  af=find_a(r2min);


  msg_printf(info,"Box extends from r= %.2lf Mpc/h to r= %.2lf Mpc/h\n",
	     sqrt(r2min),sqrt(r2max));
  msg_printf(info,"This corresponds to a in (%.2lE,%.2lE), z in (%.2lf,%.2lf)\n",
	     a0,af,1/af-1,1/a0-1);

  if(Param.a_final<af) {
    msg_printf(info,"a_final = %.2lf is unnecessary, resetting to %.2lf\n",
	       Param.a_final,af);
    Param.a_final=af;
  }
#else //_LIGHTCONE
  if(Param.a_final<Param.aout[Param.n_aout-1]) {
    msg_printf(info,"a_final = %.2lf is unnecessary, resetting to %.2lf\n",
	       Param.a_final,Param.aout[Param.n_aout-1]);
    Param.a_final=Param.aout[Param.n_aout-1];
  }
#endif //_LIGHTCONE
}
