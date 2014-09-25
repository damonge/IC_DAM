//
// COLA time integration using given force and 2LPT displacement
//
// This code is a modification to the original serial COLA code
// by Svetlin Tassev. See below.
//

/*
    Copyright (c) 2011-2013       Svetlin Tassev
                           Harvard University, Princeton University
 
    This file is part of COLAcode.

    COLAcode is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    COLAcode is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with COLAcode.  If not, see <http://www.gnu.org/licenses/>.
*/



/*

    This is COLAcode: a serial particle mesh-based N-body code 
     illustrating the COLA (COmoving Lagrangian Acceleration) method 
     described in S. Tassev, M. Zaldarriaga, D. Eisenstein (2012).
     Check that paper (refered to as TZE below) for the details. 
     Before using the code make sure you read the README file as well as
     the Warnings section below.
    
    This version: Dec 18, 2012


*/

#include <math.h>
#include <assert.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_hyperg.h> 
#include <gsl/gsl_errno.h>

#include "common.h"

static const float nLPT= -2.5f;
static const int fullT= 1; // velocity growth model

//
double Sphi(double ai, double af, double aRef);
double Sq(double ai, double af, double aRef);

// Leap frog time integration
// ** Total momentum adjustment dropped

void cola_kick(Particles* const particles,const float avel1)
{
  timer_start(evolve);  
  const float AI=particles->a_v;  // t - 0.5*dt
  const float A= particles->a_x;  // t
  const float AF=avel1;           // t + 0.5*dt

  msg_printf(normal,"Kick %g -> %g\n",AI,avel1);

  const float Om=Param.omega_m;
  const float dda=Sphi(AI,AF,A);

  const float q1=GrowthFactor(A);
  const float q2=(GrowthFactor2(A)-q1*q1);

  msg_printf(normal,"growth factor %g\n",q1);

  Particle* const P=particles->p;
  const int np=particles->np_local;
  float3* const f=particles->force;
  
  // Kick using acceleration at a= A
  // Assume forces at a=A is in particles->force

#ifdef _OPENMP
  #pragma omp parallel for default(shared)
#endif
  for(int i=0; i<np; i++) {
    float ax=-1.5*Om*dda*(f[i][0]+P[i].dx1[0]*q1+P[i].dx2[0]*q2);
    float ay=-1.5*Om*dda*(f[i][1]+P[i].dx1[1]*q1+P[i].dx2[1]*q2);
    float az=-1.5*Om*dda*(f[i][2]+P[i].dx1[2]*q1+P[i].dx2[2]*q2);

    P[i].v[0]+=ax;
    P[i].v[1]+=ay;
    P[i].v[2]+=az;
  }

  //velocity is now at a= avel1
  particles->a_v=avel1;
  timer_stop(evolve);  
}

void cola_drift(Particles* const particles,const float apos1)
{
  timer_start(evolve);
  const float A= particles->a_x; // t
  const float AC=particles->a_v; // t + 0.5*dt
  const float AF=apos1;          // t + dt

  Particle* const P=particles->p;
  const int np=particles->np_local;
  
  const float dyyy=Sq(A, AF, AC);

  const float da1=GrowthFactor(AF)-GrowthFactor(A);    // change in D_{1lpt}
  const float da2=GrowthFactor2(AF)-GrowthFactor2(A);  // change in D_{2lpt}

  msg_printf(normal,"Drift %g -> %g\n",A,AF);
    
  // Drift
#ifdef _OPENMP
  #pragma omp parallel for default(shared)
#endif
  for(int i=0;i<np;i++) {
    P[i].x[0]+=P[i].v[0]*dyyy+P[i].dx1[0]*da1+P[i].dx2[0]*da2;
    P[i].x[1]+=P[i].v[1]*dyyy+P[i].dx1[1]*da1+P[i].dx2[1]*da2;
    P[i].x[2]+=P[i].v[2]*dyyy+P[i].dx1[2]*da1+P[i].dx2[2]*da2;
  }
    
  particles->a_x=AF;
  timer_stop(evolve);
}

//
// Functions for our modified time-stepping (used when StdDA=0):
//
double gpQ(double a) { 
  return pow(a,nLPT);
}

double fun (double a,void *params) {
  double f;
  if (fullT==1) f=gpQ(a)/Qfactor(a); 
  else f=1.0/Qfactor(a);
  
  return f;
}

double Sq(double ai,double af,double aRef) {
  gsl_integration_workspace *w 
    =gsl_integration_workspace_alloc(5000);
  
  double result,error;
  double alpha=0;
  
  gsl_function F;
  F.function=&fun;
  F.params=&alpha;
  
  gsl_integration_qag(&F,ai,af,0,1e-5,5000,6,
		      w,&result,&error); 
  
  gsl_integration_workspace_free(w);
     
  if (fullT==1)
    return result/gpQ(aRef);
  return result;
}
     
double DERgpQ(double a) { // This must return d(gpQ)/da
  return nLPT*pow(a,nLPT-1);
}
     
double Sphi(double ai,double af,double aRef) {
  double result;
  result=(gpQ(af)-gpQ(ai))*aRef/Qfactor(aRef)/DERgpQ(aRef);
  
  return result;
}

// Interpolate position and velocity for snapshot at a=aout
void cola_set_LPT_snapshot(const double InitTime,Particles const * const particles,
			   Snapshot* const snapshot)
{
  timer_start(interp);
  
  const float aout=InitTime;
  const int np=particles->np_local;
  Particle const * const p=particles->p;

  ParticleMinimum* const po=snapshot->p;

  msg_printf(verbose,"Setting up inital snapshot at a= %4.2f (z=%4.2f)\n",
	     aout,1.0f/aout-1);

  const float vfac=100.0f/aout;   // km/s; H0= 100 km/s/(h^-1 Mpc)

  const float D1=GrowthFactor(aout);
  const float D2=GrowthFactor2(aout);
  const float Dv=Vgrowth(aout); // dD_{za}/dTau
  const float Dv2=Vgrowth2(aout); // dD_{2lpt}/dTau

  msg_printf(debug,"initial growth factor %5.3f %e %e\n",aout,D1,D2);
  msg_printf(debug,"initial velocity factor %5.3f %e %e\n",aout,vfac*Dv,vfac*Dv2);

#ifdef _OPENMP
  #pragma omp parallel for default(shared)  
#endif
  for(int i=0; i<np; i++) {
    Particle pa=p[i];

    po[i].v[0]=vfac*(pa.dx1[0]*Dv+pa.dx2[0]*Dv2);
    po[i].v[1]=vfac*(pa.dx1[1]*Dv+pa.dx2[1]*Dv2);
    po[i].v[2]=vfac*(pa.dx1[2]*Dv+pa.dx2[2]*Dv2);

    po[i].x[0]=pa.x[0]+D1*pa.dx1[0]+D2*pa.dx2[0];
    po[i].x[1]=pa.x[1]+D1*pa.dx1[1]+D2*pa.dx2[1];
    po[i].x[2]=pa.x[2]+D1*pa.dx1[2]+D2*pa.dx2[2];

    po[i].id=pa.id;
  }

  snapshot->np_local=np;
  snapshot->np_total=particles->np_total;
  snapshot->np_average=particles->np_average;
  snapshot->a=aout;
  timer_stop(interp);
}

// Interpolate position and velocity for snapshot at a=aout
void cola_set_snapshot(const double aout,Particles const * const particles,
		       Snapshot* const snapshot)
{
  timer_start(interp);
  const int np=particles->np_local;
  Particle const * const p=particles->p;
  float3* const f=particles->force;

  ParticleMinimum* const po=snapshot->p;
  const float Om=Param.omega_m; assert(Om>=0.0f);

  msg_printf(verbose,"Setting up snapshot at a= %4.2f (z=%4.2f) <- %4.2f %4.2f.\n",
	     aout,1.0f/aout-1,particles->a_x,particles->a_v);

  const float vfac=100.0f/aout;   // km/s; H0= 100 km/s/(h^-1 Mpc)

  const float AI=particles->a_v;
  const float A= particles->a_x;
  const float AF=aout;

  const float dda=Sphi(AI,AF,A);

  msg_printf(normal,"Growth factor of snapshot %f (a=%.3f)\n",GrowthFactor(AF),AF);

  const float q1=GrowthFactor(A);          //DAM: might be better to use 0.5*(av+aout) instead of A
  const float q2=(GrowthFactor2(A)-q1*q1); //DAM: but let's leave it for the moment

  const float Dv=Vgrowth(aout); // dD_{za}/dTau
  const float Dv2=Vgrowth2(aout); // dD_{2lpt}/dTau

  const float AC=particles->a_v;
  const float dyyy=Sq(A, AF, AC); //DAM: I'm not sure about the central value in this
                                  //DAM: and dda, but let's leave like this for the moment

  msg_printf(debug,"velocity factor %e %e\n",vfac*Dv,vfac*Dv2);
  msg_printf(debug, "RSD factor %e\n", aout/Qfactor(aout)/vfac);

  const float da1=GrowthFactor(AF)-GrowthFactor(A);    // change in D_{1lpt}
  const float da2=GrowthFactor2(AF)-GrowthFactor2(A);  // change in D_{2lpt}

#ifdef _OPENMP
  #pragma omp parallel for default(shared)  
#endif
  for(int i=0; i<np; i++) {
    // Kick + adding back 2LPT velocity + convert to km/s
    float ax=-1.5*Om*dda*(f[i][0]+p[i].dx1[0]*q1+p[i].dx2[0]*q2);
    float ay=-1.5*Om*dda*(f[i][1]+p[i].dx1[1]*q1+p[i].dx2[1]*q2);
    float az=-1.5*Om*dda*(f[i][2]+p[i].dx1[2]*q1+p[i].dx2[2]*q2);

    po[i].v[0]=vfac*(p[i].v[0]+ax+p[i].dx1[0]*Dv+p[i].dx2[0]*Dv2);
    po[i].v[1]=vfac*(p[i].v[1]+ay+p[i].dx1[1]*Dv+p[i].dx2[1]*Dv2);
    po[i].v[2]=vfac*(p[i].v[2]+az+p[i].dx1[2]*Dv+p[i].dx2[2]*Dv2);

    // Drift
    po[i].x[0]=p[i].x[0]+p[i].v[0]*dyyy+p[i].dx1[0]*da1+p[i].dx2[0]*da2;
    po[i].x[1]=p[i].x[1]+p[i].v[1]*dyyy+p[i].dx1[1]*da1+p[i].dx2[1]*da2;
    po[i].x[2]=p[i].x[2]+p[i].v[2]*dyyy+p[i].dx1[2]*da1+p[i].dx2[2]*da2;

    po[i].id=p[i].id;
  }

  snapshot->np_local=np;
  snapshot->a=aout;
  timer_stop(interp);
}

#ifdef _LIGHTCONE
#define FRAC_EXTEND_A 0.1
void cola_add_to_lightcone(Particles const * const particles,
			   Snapshot* const lcone)
{
  float amax=-1,amin=10000,dum;

  timer_start(interp);
  const int np=particles->np_local;
  Particle *p=particles->p;
  float3* const f=particles->force;
  int np_updated=lcone->np_local;

  ParticleMinimum* const po=lcone->p;
  const float Om=Param.omega_m; assert(Om>=0.0f);

  const float AI=particles->a_v;
  const float A= particles->a_x;
  const float AC=particles->a_v;
  const float gf1=GrowthFactor(A);
  const float gf2=GrowthFactor2(A);
  const float q1=GrowthFactor(A);
  const float q2=(GrowthFactor2(A)-q1*q1);

  if(AI>amax) amax=AI;
  if(AI<amin) amin=AI;
  if(A>amax) amax=A;
  if(A<amin) amin=A;
  //Extend a little bit to avoid missing particles
  dum=(amax-amin)*FRAC_EXTEND_A;
  if(amax+dum<=1) amax+=dum;
  else amax=1;
  if(amin-dum>=0) amin-=dum;

  msg_printf(verbose,"Storing particles in the lighcone between times %4.2f %4.2f\n",amin,amax);

#ifndef _PROPER_LC_INTERPOLATION
  //Should we do a proper interpolation?
  float a_interval=amax-amin;
  float dda_0=Sphi(AI,amin,A);
  float dda_tilt=(Sphi(AI,amax,A)-dda_0)/a_interval;
  float Dv_0=Vgrowth(amin);
  float Dv_tilt=(Vgrowth(amax)-Dv_0)/a_interval;
  float Dv2_0=Vgrowth2(amin);
  float Dv2_tilt=(Vgrowth2(amax)-Dv2_0)/a_interval;
  float dyyy_0=Sq(A,amin,AC);
  float dyyy_tilt=(Sq(A,amax,AC)-dyyy_0)/a_interval;
  float da1_0=GrowthFactor(amin)-gf1;
  float da1_tilt=(GrowthFactor(amax)-gf1-da1_0)/a_interval;
  float da2_0=GrowthFactor2(amin)-gf2;
  float da2_tilt=(GrowthFactor2(amax)-gf2-da2_0)/a_interval;
#endif //_PROPER_LC_INTERPOLATION
  //Need a(r) for vfac, dda, Dv, Dv2, dyyy, da1, da2

  int db_n_stored=0;
  int db_n_lost=0;
  for(int i=0;i<np;i++) {
    if(p[i].in_lc==0) {
      float aout=x2a(p[i].x);
      
      if(amin<=aout && aout<=amax){
#ifndef _PROPER_LC_INTERPOLATION
	double d_a=aout-amin;
	float dda=dda_0+dda_tilt*d_a;
	float Dv=Dv_0+Dv_tilt*d_a;
	float Dv2=Dv2_0+Dv2_tilt*d_a;
	float dyyy=dyyy_0+dyyy_tilt*d_a;
	float da1=da1_0+da1_tilt*d_a;
	float da2=da2_0+da2_tilt*d_a;
#else //_PROPER_LC_INTERPOLATION
	float dda=Sphi(AI,aout,A);
	float Dv=Vgrowth(aout);
	float Dv2=Vgrowth2(aout);
	float dyyy=Sq(A,aout,AC);
	float da1=GrowthFactor(aout)-gf1;
	float da2=GrowthFactor2(aout)-gf2;
#endif //_PROPER_LC_INTERPOLATION

	float vfac=100.0f/aout;

	float ax=-1.5*Om*dda*(f[i][0]+p[i].dx1[0]*q1+p[i].dx2[0]*q2);
	float ay=-1.5*Om*dda*(f[i][1]+p[i].dx1[1]*q1+p[i].dx2[1]*q2);
	float az=-1.5*Om*dda*(f[i][2]+p[i].dx1[2]*q1+p[i].dx2[2]*q2);
	
	po[np_updated].v[0]=vfac*(p[i].v[0]+ax+p[i].dx1[0]*Dv+p[i].dx2[0]*Dv2);
	po[np_updated].v[1]=vfac*(p[i].v[1]+ay+p[i].dx1[1]*Dv+p[i].dx2[1]*Dv2);
	po[np_updated].v[2]=vfac*(p[i].v[2]+az+p[i].dx1[2]*Dv+p[i].dx2[2]*Dv2);
	
	po[np_updated].x[0]=p[i].x[0]+p[i].v[0]*dyyy+p[i].dx1[0]*da1+p[i].dx2[0]*da2;
	po[np_updated].x[1]=p[i].x[1]+p[i].v[1]*dyyy+p[i].dx1[1]*da1+p[i].dx2[1]*da2;
	po[np_updated].x[2]=p[i].x[2]+p[i].v[2]*dyyy+p[i].dx1[2]*da1+p[i].dx2[2]*da2;
	
	po[np_updated].id=p[i].id;
	
	p[i].in_lc=1;

	np_updated++;
	db_n_stored++;
      }
      else if(aout<=amax) {
	//	printf("db ----- shit, lost particle %lE %lE %lE %lld\n",amin,amax,aout,p[i].id);
	db_n_lost++;
      }
    }
  }

  printf("Stored %d, %d so far\n",db_n_stored,np_updated);
  printf("Lost %d (%.2lE%%) so far\n",db_n_lost,100*(double)db_n_lost/np);

  lcone->np_local=np_updated;
  timer_stop(interp);
}

void cola_add_to_lightcone_LPT(const float a_init,
			       Particles const * const particles,
			       Snapshot* const lcone)
{
  timer_start(interp);
  const int np=particles->np_local;
  Particle *p=particles->p;
  int np_updated=lcone->np_local;
  
  ParticleMinimum* const po=lcone->p;
  const float Om=Param.omega_m; assert(Om>=0.0f);
  float gf1_0=GrowthFactor(a_init);
  float gf2_0=GrowthFactor2(a_init);

  msg_printf(verbose,"Storing particles in the lighcone before initial condition (a=%4.2f)\n",a_init);

  for(int i=0;i<np;i++) {
    if(p[i].in_lc==0) {
      float aout=x2a(p[i].x);
      
      if(aout<=a_init) {
	float vfac=100.0f/aout;
	float dgf1=GrowthFactor(aout)-gf1_0;
	float dgf2=GrowthFactor2(aout)-gf2_0;
	float Dv1=Vgrowth(aout);
	float Dv2=Vgrowth2(aout);

	po[np_updated].x[0]=p[i].x[0]+p[i].dx1[0]*dgf1+p[i].dx2[0]*dgf2;
	po[np_updated].x[1]=p[i].x[1]+p[i].dx1[1]*dgf1+p[i].dx2[1]*dgf2;
	po[np_updated].x[2]=p[i].x[2]+p[i].dx1[2]*dgf1+p[i].dx2[2]*dgf2;

	po[np_updated].v[0]=vfac*(p[i].dx1[0]*Dv1+p[i].dx2[0]*Dv2);
	po[np_updated].v[1]=vfac*(p[i].dx1[1]*Dv1+p[i].dx2[1]*Dv2);
	po[np_updated].v[2]=vfac*(p[i].dx1[2]*Dv1+p[i].dx2[2]*Dv2);

	po[np_updated].id=p[i].id;
	
	p[i].in_lc=1;

	np_updated++;
      }
    }
  }

  lcone->np_local=np_updated;
  timer_stop(interp);
}
#endif //_LIGHTCONE
