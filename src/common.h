#ifndef _COMMON_H_
#define _COMMON_H_

#include <mpi.h>

//////
// Defined in parameters.h
typedef struct {
  int nc;

  double np_alloc_factor;
  double boxsize;
  int nbox_per_side;

  double omega_m, sigma8, h;

  int random_seed;

  double a_final;

  char power_spectrum_filename[256];
  char init_filename[256];
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
#ifdef _LONGIDS
  unsigned long long id;
#else //_LONGIDS
  unsigned int id;
#endif //_LONGIDS
} Particle;

//////
// Defined in comm.h
void comm_init(const int nc_part,const float boxsize);
int comm_this_node(void);
int comm_nnode(void);


//////
// Defined in cosmo.h
double PowerSpec(const double k);
void cosmo_init(const char filename[],const double sigma8,
		const double omega_m,const double omega_lambda,
		const double a_initial);
double GrowthFactor(const double a);
double GrowthFactor2(const double a);
double Vgrowth(const double a);
double Vgrowth2(const double a);


//////
// Defined in lpt.h
void lpt_init(const int nc);
void lpt_end(void);
void lpt_set_displacement(const int Seed,const double Box);


//////
// Defined in msg.h
void msg_init(void);
void msg_printf(const char *fmt, ...);
void msg_abort(const int errret,const char *fmt, ...);


//////
// Defined in timer.h
enum Category {Init,LPT,Snp};
enum SubCategory {all};

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
  int flag_gadgetformat;
  char fill[56];
} GadgetHeader;

void write_snapshot(const char filebase[],const double a_init);
void write_snapshot_cola(const char filebase[],const double a_init);

#endif //_COMMON_H_
