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

static int NPowerTable;
static double r_tophat;
static double Omega, OmegaLambda;

static struct pow_table
{
  double logk, logD;
} *PowerTable;

static void read_power_table_camb(const char filename[])
{
  double k, p;
  double fac= 1.0/(2.0*M_PI*M_PI);

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

  msg_printf("Found %d pairs of values in input spectrum table\n",NPowerTable);

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

  double P=Delta2/(4.0*M_PI*k*k*k);

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

static void normalize_power(const double sigma8)
{
  //DAM: this only checks that the actual sigma8 is right
  const double R8=8.0; // 8 Mpc

  double res=TopHatSigma2(R8); 
  double sigma8_input=sqrt(res);

  if(abs(sigma8_input-sigma8)/sigma8>0.05)
    msg_abort(3010,"Input sigma8 %f is far from target sigma8 %f\n",
	      sigma8_input,sigma8);

  msg_printf("Input power spectrum sigma8 %f\n",sigma8_input);
}

static double omega_a(const double a)
{
  return Omega/(Omega+(1-Omega-OmegaLambda)*a+
		 OmegaLambda*a*a*a);
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

static double Qfactor(const double a)
{
  double acube=a*a*a;
  return sqrt(Omega*acube+OmegaLambda*acube*acube+
	      (1-Omega-OmegaLambda)*acube*a);
}

double GrowthFactor(const double a)
{
  return growth(a)/growth(1.0);
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

void cosmo_init(const char filename[],const double sigma8,
		const double omega_m,const double omega_lambda,
		const double a_initial)
{
  Omega=omega_m;
  OmegaLambda=omega_lambda;

  int myrank=comm_this_node();

  if(myrank==0) {
    read_power_table_camb(filename);
    normalize_power(sigma8);
  }

  msg_printf("Powerspectrum file: %s\n",filename);
  msg_printf("Growth at a=%lE is %lE\n",a_initial,GrowthFactor(a_initial));

  MPI_Bcast(&NPowerTable,1,MPI_INT,0,MPI_COMM_WORLD);
  if(myrank!=0) {
    PowerTable=malloc(NPowerTable*sizeof(struct pow_table));
  }

  MPI_Bcast(PowerTable,NPowerTable*sizeof(struct pow_table),MPI_BYTE,0,
	    MPI_COMM_WORLD);
}
