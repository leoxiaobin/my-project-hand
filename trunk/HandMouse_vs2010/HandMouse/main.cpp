#include "Bspline.h"
#include "Hand.h"
#include "opencv.h"
#include "TSLskinSeg.h"
#include "particles.h"
#include "Function.h"
#include <ctime>

string confile = "config4.txt";
string cntpntfile = "controlPoint2.txt";
string mlinefile = "measuremenline.txt";

string xlm = "test.xml";

const int NUM_PARTICLES = 500;


int main()
{
	Hand originalHand(confile, cntpntfile, mlinefile);
	originalHand.affineHand(0, -50, 0.6, 0);

	cv::VideoCapture capture(0);

	if (!capture.isOpened())
	{
		std::cerr<<"Can't open the video camera!"<<std::endl;
		return -1;
	}

	cv::Mat img;
	capture>>img;

	int imgWidth = img.cols;
	int imgHeight = img.rows;
	cv::Mat BW = Mat::zeros(2*imgWidth, 2*imgHeight, CV_8UC1);
	cv::Mat imgBw(BW, cv::Rect(0.5*imgWidth, 0.5*imgHeight, imgWidth, imgHeight));

	particle preParticles;
	particle *particles, *newParticles;

	preParticles.w = 0;
	preParticles.x = originalHand.GetGravity().x;
	preParticles.y = originalHand.GetGravity().y;
	preParticles.s = 1;
	preParticles.a = 0;

	particles = init_distribution(originalHand.GetGravity(), NUM_PARTICLES);

	int x(0), y(0);
	float s(1.0), a(0.0);

	cv::RNG rng(time(NULL));

	bool keep = false;

	while (1)
	{
		capture>>img;
		TSLskinSegment(img,imgBw);
		medianBlur(imgBw,imgBw,7);
	
		originalHand.showHand(img,Scalar(255,255,0),3);
		//hand.showMeasureLinePoints(img, cv::Scalar(255,0,255),2);
		cv::flip(img,img,1);
		imshow("result",img);
	
		if (waitKey(20)>=0)
		{
			break;
		}
		if(originalHand.calWeight(imgBw)>1) break;
	}


	while(1)
	{
		capture>>img;
		TSLskinSegment(img, imgBw);
		medianBlur(imgBw, imgBw, 11);
		imshow("bw", imgBw);

		if (!keep)
		{
#pragma omp parallel for
			for (int i = 0; i<NUM_PARTICLES; ++i)
			{
				Hand trasitHand(originalHand);

				particles[i] = transition(particles[i], imgWidth, imgHeight, rng);

				x = cvRound(particles[i].x - originalHand.GetGravity().x);
				y = cvRound(particles[i].y - originalHand.GetGravity().y);
				s = particles[i].s;
				a = particles[i].a;

				a = 180*a/CV_PI;

				trasitHand.affineHand(x, y, s, a);
				particles[i].w = trasitHand.calWeight(imgBw);
				//particles[i].w = trasitHand.calPalmWeight(imgBw);

			}
		}

		Hand preHand(originalHand);
		x = cvRound(preParticles.x - originalHand.GetGravity().x);
		y = cvRound(preParticles.y - originalHand.GetGravity().y);
		s = preParticles.s;
		a = preParticles.a;

		a = 180*a/CV_PI;

		preHand.affineHand(x, y, s, a);

		preParticles.w = preHand.calWeight(imgBw);
		//preParticles.w = preHand.calPalmWeight(imgBw);

		qsort(particles, NUM_PARTICLES, sizeof(particle), &particle_cmp);

		if (preParticles.w < particles[0].w)
		{
			preParticles = particles[0];

			normalize_weights(particles, NUM_PARTICLES);

			newParticles = resample(particles, NUM_PARTICLES);

			free(particles);
			particles = newParticles;

			keep = false;
		} 
		else
		{
			keep = true;
		}

		x = cvRound(preParticles.x - originalHand.GetGravity().x);
		y = cvRound(preParticles.y - originalHand.GetGravity().y);
		s = preParticles.s;
		a = preParticles.a;

		a = 180*a/CV_PI;

		Hand showHand(originalHand);
		showHand.affineHand(x, y, s, a);

		sweepFinger(showHand, Hand::INDEX, -30, imgBw);
		refineFinger(showHand, Hand::INDEX, imgBw);

		sweepFinger(showHand, Hand::MIDDLE, -20, imgBw);
		refineFinger(showHand, Hand::MIDDLE, imgBw);

		sweepFinger(showHand, Hand::RING, -20, imgBw);
		refineFinger(showHand, Hand::RING, imgBw);

		sweepFinger(showHand, Hand::LITTLE, -20, imgBw);
		refineFinger(showHand, Hand::LITTLE, imgBw);

		sweepThumb1(showHand, imgBw);
		sweepThumb2(showHand, imgBw);

		//showHand.affineFinger(Hand::THUMB1, 1, 20);
		//showHand.affineThumb1(90);
		showHand.showHand(img, cv::Scalar(0.255,111),2);
		//showHand.showControlPoints(img,cv::Scalar(255,0,0),6);


		cv::flip(img,img,1);
		cv::imshow("result", img);

		if (waitKey(20)>=0)
		{
			break;
		}	
	}

	return 0;
}