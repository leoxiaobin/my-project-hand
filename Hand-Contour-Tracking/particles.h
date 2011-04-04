#ifndef PARTICLES_H
#define PARTICLES_H
#include <string>
#include "opencv.h"
/******************************* Definitions *********************************/

/* standard deviations for gaussian sampling in transition model */

/* autoregressive dynamics parameters for transition model */
//const float A1=1.5733; 
//const float A2=-0.6188;
//const float Bx =9.2498;
//const float By=8.3248;
const float Ba=0.0555;
//const float Bs=0.0185;
//const float PI=3.1415;

/* standard deviations for gaussian sampling in transition model */
#define TRANS_X_STD 15.0
#define TRANS_Y_STD 15.0
#define TRANS_S_STD 0.01

/* autoregressive dynamics parameters for transition model */
#define A1  1.2
#define A2 -0.2
#define B0  1.0000

typedef struct particle {
  float x;          /**< current x coordinate */
  float y;          /**< current y coordinate */
  float a;
  float s;          /**< scale */

  float xp;         /**< previous x coordinate */
  float yp;         /**< previous y coordinate */
  float sp;         /**< previous scale */
  float ap;

  float w;          /**< weight */
} particle;


particle* init_distribution(cv::Point2f center, int p);

particle transition( particle p, int w, int h, cv::RNG& rng );

void normalize_weights( particle* particles, int n );

particle* resample( particle* particles, int n );

int particle_cmp( const void* p1, const void* p2 );


#endif
