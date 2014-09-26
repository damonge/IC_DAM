#ifndef _COMMON_H_
#define _COMMON_H_

#include <mpi.h>

//////
// Defined in parameters.h
typedef struct {
  int nc;

  double np_alloc_factor;
  double boxsize;

  double omega_m, sigma8, h;

  int random_seed;

  double a_final;

  char power_spectrum_filename[256];
  char init_filename[256];

  int loglevel;
} Parameters;
Parameters Param;

int read_parameters(char *fname);


//////
// Defined in particle.h
typedef float float3[3];

typedef struct {
  float x[3];
  float v[3];
  float dx1[3];
  float dx2[3];
  long long id;
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
// Defined in comm.h
void comm_init(const int nc_part,const float boxsize);
int comm_this_node(void);
int comm_nnode(void);


//////
// Defined in cosmo.h
double PowerSpec(const double k);
void cosmo_init(const char filename[],const double sigma8,
		const double omega_m,const double omega_lambda);
double GrowthFactor(const double a);
double GrowthFactor2(const double a);
double Vgrowth(const double a);
double Vgrowth2(const double a);


//////
// Defined in lpt.h
void lpt_init(const int nc, const void* mem, const size_t size);
void lpt_set_displacement(const int Seed,const double Box,
			  const double a_init,Snapshot* const snapshot);
int lpt_get_local_nx(void);


//////
// Defined in mem.h
typedef struct {
  void *mem1, *mem2;
  size_t size1, size2;
} Memory;
Snapshot* allocate_snapshot(const int nc, const int nx,const double np_alloc_factor,
			    void* const mem,const size_t mem_size);
void allocate_shared_memory(const int nc,const double np_alloc_factor,
			    Memory* const mem);


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

#endif //_COMMON_H_
