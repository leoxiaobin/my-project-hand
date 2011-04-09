#include "Bspline.h"
#include "Hand.h"
#include "opencv.h"
#include "TSLskinSeg.h"
#include "particles.h"
#include "Function.h"
#include "CbCrSelfAdaptSkinDetect.h"
#include <ctime>

string confile = "config4.txt";
string cntpntfile = "controlPoint2.txt";
string mlinefile = "measuremenline.txt";

string xlm = "test.xml";

const int NUM_PARTICLES = 1000;


int main()
{
	Hand originalHand(confile, cntpntfile, mlinefile);

	cv::Mat testImg = cv::imread("ii.jpg");
	originalHand.showHand(testImg, cv::Scalar(23,21,222),2);
	originalHand.showMeasureLinePoints(testImg, cv::Scalar(211,2,2),2);

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
	cv::Mat BW = Mat::zeros(2*imgHeight, 2*imgWidth, CV_8UC1);
	cv::Mat imgBw(BW, cv::Rect(0.5*imgWidth, 0.5*imgHeight, imgWidth, imgHeight));
	cv::Mat mask = Mat::zeros(imgHeight, imgWidth, CV_8UC1);

	setMask(originalHand, mask);
	cv::flip(mask, mask, 1);

	imshow("mask", mask);

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


	CbCrSelfAdaptSkinDetect skinDetector;
	int countFit = 0;
	double t;


	while (1)
	{
		capture>>img;
		TSLskinSegment(img,imgBw);
		medianBlur(imgBw,imgBw,7);
	
		originalHand.showHand(img,Scalar(255,255,0),3);
		//originalHand.showMeasureLinePoints(img, cv::Scalar(255,0,255),2);
		cv::flip(img,img,1);
		imshow("result",img);
	
		if (waitKey(20)>=0)
		{
			break;
		}
		if(originalHand.calWeight(imgBw)>20) 
		{
			Mat _img, _mask;
			cv::resize(img, _img, cv::Size(160, 120));
			cv::resize(mask, _mask, cv::Size(160, 120));
			t= (double)getTickCount();
			skinDetector.getBestParamentMask(_img, _mask);
			t = (double)getTickCount() - t;
			printf("detection time = %gms\n", t*1000./cv::getTickFrequency());
			countFit++;
		}

		if(countFit == 20) break;
	}


	while(1)
	{
		capture>>img;
		t = (double)getTickCount();
		//TSLskinSegment(img, imgBw);
		skinDetector.skinDetectForUser(img, imgBw);
		medianBlur(imgBw, imgBw, 5);
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
		showHand.showHand(img, cv::Scalar(0.255,255),2);
		//showHand.showMeasureLinePoints(img, cv::Scalar(255,0,0),2);
		//showHand.showControlPoints(img,cv::Scalar(255,0,0),6);
		//showHand.showMeasurePoints(img, cv::Scalar(0,255,0),3);

		t = (double)getTickCount() - t;
		printf("detection time = %gms\n", t*1000./cv::getTickFrequency());


		cv::flip(img,img,1);
		cv::imshow("result", img);

		if (waitKey(20)>=0)
		{
			break;
		}	
	}
	return 0;
}