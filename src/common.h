#ifndef _COMMON_H_
#define _COMMON_H_

#include <mpi.h>

//////
// Defined in parameters.h
typedef struct {
  int nc;
  int pm_nc_factor;

  double np_alloc_factor;
  double boxsize;

  double omega_m, sigma8, h;

  int random_seed;

  int ntimestep;
  double a_final;
  int n_aout;
  double* aout;

#ifdef _LIGHTCONE
  float pos_obs[3];
#endif //_LIGHTCONE

  char power_spectrum_filename[256];
  char snap_filename[256];
  char init_filename[256];
  char dens_filename[256];
  char disp_filename[256];

  int loglevel;

  int write_init;
  int write_snap;
  int write_dens;
  int write_disp;

} Parameters;
Parameters Param;

int read_parameters(char *fname);


//////
// Defined in particle.h
typedef float float3[3];

typedef struct {
  float x[3];
  float dx1[3]; // ZA displacement
  float dx2[3]; // 2LPT displacement
  float v[3];   // velocity
  long long id;
#ifdef _LIGHTCONE
  char in_lc;
#endif //_LIGHTCONE
} Particle;

typedef struct {
  Particle* p;
  float3* force;
  float a_x, a_v;

  int np_local, np_allocated;
  long long np_total;
  float np_average;
} Particles;

typedef struct {
  float x[3];
  float v[3];
  long long id;
#ifdef _LIGHTCONE
  float a;
#endif //_LIGHTCONE
} ParticleMinimum;

typedef struct {
  ParticleMinimum* p;
  int np_local;
  int np_allocated;
  long long np_total;
  float np_average;
  float a;
  int nc;
} Snapshot;


//////
// Defined in cola.h
void cola_kick(Particles* const particles,const float avel1);
void cola_drift(Particles* const particles,const float apos1);
void cola_set_LPT_snapshot(const double InitTime,Particles const * const particles,
			   Snapshot* const snapshot);
void cola_set_snapshot(const double aout,Particles const * const particles,
		       Snapshot* const snapshot);
#ifdef _LIGHTCONE
void cola_add_to_lightcone(Particles const * const particles,Snapshot* const lcone);
void cola_add_to_lightcone_LPT(const float a_init,Particles const * const particles,
			       Snapshot* const lcone);
#endif //_LIGHTCONE


//////
// Defined in comm.h
enum Direction {ToRight=0, ToLeft=1};

void comm_init(const int nc_pm,const int nc_part,const float boxsize);
int comm_this_node(void);
int comm_reduce_int(int x,MPI_Op op);
int comm_share_int(int x,MPI_Op op);
float comm_xleft(const int dix);
float comm_xright(const int dix);
int comm_node(const int dix);
int comm_nnode(void);
// These don't seem to be needed
//float comm_xmax(void);
//float comm_xmin(void);
//int comm_right_edge(void);
//int comm_get_nrecv(enum Direction direction,int nsend);
//void comm_sendrecv(enum Direction direction,void* send,int nsend,
//		   void* recv,int nrecv,MPI_Datatype datatype);
//int comm_get_total_int(int x);


//////
// Defined in cosmo.h
double PowerSpec(const double k);
void cosmo_init(const char filename[],const double sigma8,
		const double omega_m,const double omega_lambda);
double GrowthFactor(const double a);
double GrowthFactor2(const double a);
double Vgrowth(const double a);
double Vgrowth2(const double a);
double Qfactor(const double a);
float x2a(float const * const x);
void set_a_final(void);


//////
// Defined in lpt.h
void lpt_init(const int nc, const void* mem, const size_t size);
void lpt_set_displacement(const int Seed,const double Box,Particles* const particles);
void lpt_set_initial_condition(const double InitTime,
			       const double Box,Particles* const particles);
int lpt_get_local_nx(void);


//////
// Defined in pm.h
void pm_init(const int nc_pm,const int nc_pm_factor,const float boxsize,
	     void* const mem1,const size_t size1,
	     void* const mem2,const size_t size2);
void pm_calculate_forces(Particles*);
void pm_write_density(const char fname_base[],Snapshot* snap);


//////
// Defined in move.h
void move_particles2(Particles* const particles, const float BoxSize,
		     void* const buf, const size_t size);
void move_particles_snap(Snapshot* const particles, const float BoxSize,
			 void* const buf, const size_t size);


//////
// Defined in mem.h
typedef struct {
  void *mem1, *mem2;
  size_t size1, size2;
} Memory;

Particles* allocate_particles(const int nc, const int nx, double np_alloc_factor);
#ifdef _LIGHTCONE
Snapshot *allocate_lightcone(const int nc,const int nx,const double np_alloc_factor);
#endif //_LIGHTCONE
Snapshot* allocate_snapshot(const int nc, const int nx,const int np_alloc,
			    void* const mem,const size_t mem_size);
void allocate_shared_memory(const int nc, const int nc_factor,
			    const double np_alloc_factor, Memory* const mem);


//////
// Defined in msg.h
enum LogLevel {verbose,debug,normal,info,warn,error,fatal,silent};

void msg_init(void);
void msg_set_loglevel(const enum LogLevel log_level);
void msg_printf(const enum LogLevel level,const char *fmt, ...);
void msg_abort(const int errret,const char *fmt, ...);


//////
// Defined in timer.h
enum Category {Init,LPT,COLA,Snp};
enum SubCategory {all,fft,assign,force_mesh,pforce,check,
		  comm,evolve,write,kd_build,kd_link,
		  interp,global,sub};

void timer_set_category(enum Category new_cat);
void timer_start(enum SubCategory sub);
void timer_stop(enum SubCategory sub);
void timer_print();


//////
// Defined in write.h
typedef struct {
  int    np[6];
  double mass[6];
  double time;
  double redshift;
  int    flag_sfr;
  int    flag_feedback;
  unsigned int np_total[6];
  int    flag_cooling;
  int    num_files;
  double boxsize;
  double omega0;
  double omega_lambda;
  double hubble_param; 
  int flag_stellarage;
  int flag_metals;
  unsigned int np_total_highword[6];
  int  flag_entropy_instead_u;
  char fill[60];
} GadgetHeader;

void write_snapshot(const char filebase[],Snapshot const * const snapshot);
void write_force(const char filebase[],Particles const * const particles);
void write_displacements(const char filebase[],Particles const * const particles);

#endif //_COMMON_H_
