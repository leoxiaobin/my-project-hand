#include <iostream>
#include "particles.h"
#include "opencv.h"
#include <string>

particle* init_distribution( cv::Point2f center, int p)
{
  particle* particles;
  float x, y;
  int k = 0;
  x = center.x;
  y = center.y;
  
  particles = (particle *)malloc( p * sizeof( particle ) );
  

  /* create particles at the centers of each of n regions */
  for( k = 0; k < p; k++ )
  {
	  particles[k].xp = particles[k].x = x;
	  particles[k].yp = particles[k].y = y;
	  particles[k].sp = particles[k].s = 1.0;
	  particles[k].ap = particles[k].a = 0;
	  particles[k].w = 0;
  }

  /* make sure to create exactly p particles */
  while( k < p )
  {
	  particles[k].xp = particles[k].x = x;
	  particles[k].yp = particles[k].y = y;
	  particles[k].sp = particles[k].s = 1.0;
	  particles[k].ap = particles[k].a = 0;
	  particles[k].w = 0;
  }

  return particles;
}


particle transition( particle p, int w, int h, cv::RNG& rng )
{
  float x, y, s;
  particle pn;
 

  x = A1 * ( p.x ) + A2 * ( p.xp ) +
	  B0 * rng.gaussian( TRANS_X_STD ) ;
  pn.x = MAX( 0.0, MIN( (float)w - 1.0, x ) );
  y = A1 * ( p.y ) + A2 * ( p.yp ) +
	  B0 * rng.gaussian( TRANS_Y_STD ) ;
  pn.y = MAX( 0.0, MIN( (float)h - 1.0, y ) );
  s = A1 * ( p.s - 1.0 ) + A2 * ( p.sp - 1.0 ) +
	  B0 * rng.gaussian( TRANS_S_STD ) + 1.0;
  pn.s = MAX( 0.1, s );

  pn.a = A1 * p.a + A2 * p.ap +
   Ba *rng.gaussian( 1 );




  ///* sample new state using second-order autoregressive dynamics */
  //x = A1 *  p.x  + A2 *  p.xp  +
	 // Bx * rng.gaussian(1);
  //pn.x = MAX( 0.0, MIN( (float)w - 1.0, x ) );
  //y = A1 *  p.y  + A2 *  p.yp  +
	 // By * rng.gaussian(1);
  //pn.y = MAX( 0.0, MIN( (float)h - 1.0, y ) );
  //float tmp1 = rng.gaussian(1);
  //s = 1  +
	 // 0.0116 * tmp1;
  //pn.s = MAX( 0.1, s );
  //float tmp = rng.gaussian(1);
  //std::cout<<"s = "<<p.s<<std::endl;
  //std::cout<<"sp = "<<p.sp<<std::endl;
  //std::cout<<"gaussian = "<<tmp1<<std::endl;
  //std::cout<<"sn = "<<s<<std::endl;
  //pn.a = A1 * p.a + A2 * p.ap +
	 // Ba *tmp;
  //std::cout<<"ainside = "<<pn.a<<std::endl;
  //pn.a = a % (2*PI);
  pn.xp = p.x;
  pn.yp = p.y;
  pn.sp = p.s;
  pn.ap = p.a;

  //pn.x0 = p.x0;
  //pn.y0 = p.y0;
  //pn.s0 = p.s0;
  //pn.a0 = p.a0;

  pn.w = 0;

  return pn;
}


void normalize_weights( particle* particles, int n )
{
  float sum = 0;
  int i;

  for( i = 0; i < n; i++ )
    sum += particles[i].w;
  for( i = 0; i < n; i++ )
    particles[i].w /= sum;
}


particle* resample( particle* particles, int n )
{
  particle* new_particles;
  int i, j, np, k = 0;

  //qsort( particles, n, sizeof( particle ), &particle_cmp );
  new_particles = (particle *)malloc( n * sizeof( particle ) );
  for( i = 0; i < n; i++ )
    {
      np = cvRound( particles[i].w * n );
      for( j = 0; j < np; j++ )
	{
	  new_particles[k++] = particles[i];
	  if( k == n )
	    goto exit;
	}
    }
  while( k < n )
    new_particles[k++] = particles[0];

 exit:
  return new_particles;
}


int particle_cmp( const void* p1, const void* p2 )
{
  particle* _p1 = (particle*)p1;
  particle* _p2 = (particle*)p2;

  if( _p1->w > _p2->w )
    return -1;
  if( _p1->w < _p2->w )
    return 1;
  return 0;
}


