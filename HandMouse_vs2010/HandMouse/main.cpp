#include "Bspline.h"
#include "Hand.h"
#include "opencv.h"
#include "TSLskinSeg.h"
#include "particles.h"
#include "Function.h"
#include "CbCrSelfAdaptSkinDetect.h"
#include <ctime>

const string confile = "config4.txt";
const string cntpntfile = "controlPoint2.txt";
const string mlinefile = "measuremenline.txt";

const string xlm = "test.xml";

const string hogXlm = "c:\\OpenCV\\HOG\\hog_64x64_16x32_8x16_8x16_2.xml";

const int NUM_PARTICLES = 2000;

const float THRESHOLD = 1e5;


int main()
{
	Hand originalHand(confile, cntpntfile, mlinefile);
	HOGDescriptor HandClassifier("c:\\OpenCV\\HOG\\hog_64x64_16x32_8x16_8x16_2.xml");

	//Hand originalHand("newhand.xml");

	originalHand.affineHand(0, -50, 0.6, 0);
	//originalHand.saveXML("newhand.xml");

	cv::Rect handROI;

	vector<cv::Rect> foundHands;



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

	//imshow("mask", mask);

	particle preParitcle;
	particle *particles, *newParticles;

	//preParticles.w = 0;
	//preParticles.x = originalHand.GetGravity().x;
	//preParticles.y = originalHand.GetGravity().y;
	//preParticles.s = 1;
	//preParticles.a = 0;

	//particles = init_distribution(originalHand.GetGravity(), NUM_PARTICLES);

	int x(0), y(0);
	float s(1.0), a(0.0);

	cv::RNG rng(time(NULL));


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
		if(originalHand.calWeight(imgBw)>100) 
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

		if(countFit == 2) break;
	}

	cv::Mat element = cv::Mat::ones(3,3,CV_8UC1);
	vector< vector<cv::Point> > contours;
	vector< Vec4i> hierarchy;

	int countDetectHand(0), countTracking(0), countWeight(0);

	bool detect = true;
	bool keep = false;

	Hand hand(originalHand);

	int handCenterX(0), handCenterY(0);

	while(1)
	{
		capture>>img;
		Mat imgCopy;
		img.copyTo(imgCopy);
		if (detect)
		{
			HandClassifier.detectMultiScale(imgCopy,foundHands,1,cv::Size(8,8), cv::Size(0,0), 1.12, 2);
			if (foundHands.size() != 0)
			{
				countDetectHand++;
			} 
			else
			{
				countDetectHand = 0;
			}

			
			if (countDetectHand == 3)
			{
				handROI = foundHands[0];

				countDetectHand = 0;

				detect = false;

				cv::Point2f moveHand;
				moveHand.x = handROI.x - originalHand.GetGravity().x + 0.5*handROI.width;
				moveHand.y = handROI.y - originalHand.GetGravity().y + 0.5*handROI.height;

				float handScale = 1;

				handScale = 2.0*(float)handROI.area()/(float)originalHand.GetArea();

				hand = originalHand;

				hand.affineHand(moveHand.x, moveHand.y, handScale, 0);
				preParitcle.w = 0;
				preParitcle.x = hand.GetGravity().x;
				preParitcle.y = hand.GetGravity().y;
				preParitcle.s = 1;
				preParitcle.a = 0;

				particles = init_distribution(hand.GetGravity(), NUM_PARTICLES);

			}
			foundHands.clear();

			cv::waitKey(10);
		} 
		else
		{
			skinDetector.skinDetectForUser(img, imgBw);
			cv::morphologyEx(imgBw, imgBw, MORPH_OPEN, element,cv::Point(-1,-1), 1);
			cv::morphologyEx(imgBw, imgBw, MORPH_CLOSE, element,cv::Point(-1,-1), 1);
			cv::findContours(imgBw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			cv::drawContours(imgBw,contours,-1,255,CV_FILLED,8,hierarchy,2);
			imshow("bw", imgBw);

			if (!keep)
			{
#pragma omp parallel for
				for (int i = 0; i<NUM_PARTICLES; ++i)
				{
					Hand trasitHand(hand);

					particles[i] = transition(particles[i], imgWidth, imgHeight, rng);

					x = cvRound(particles[i].x - hand.GetGravity().x);
					y = cvRound(particles[i].y - hand.GetGravity().y);
					s = particles[i].s;
					a = particles[i].a;

					a = 180*a/CV_PI;

					trasitHand.affineHand(x, y, s, a);
					particles[i].w = trasitHand.calWeight(imgBw);
					//particles[i].w = trasitHand.calPalmWeight(imgBw);
				}
			}

			Hand preHand(hand);
			x = cvRound(preParitcle.x - originalHand.GetGravity().x);
			y = cvRound(preParitcle.y - originalHand.GetGravity().y);
			s = preParitcle.s;
			a = preParitcle.a;

			a = 180*a/CV_PI;

			preHand.affineHand(x, y, s, a);

			preParitcle.w = preHand.calWeight(imgBw);
			//preParticles.w = preHand.calPalmWeight(imgBw);

			qsort(particles, NUM_PARTICLES, sizeof(particle), &particle_cmp);

			if (preParitcle.w < particles[0].w)
			{
				preParitcle = particles[0];

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

			x = cvRound(preParitcle.x - hand.GetGravity().x);
			y = cvRound(preParitcle.y - hand.GetGravity().y);
			s = preParitcle.s;
			a = preParitcle.a;

			a = 180*a/CV_PI;

			Hand showHand(hand);
			showHand.affineHand(x, y, s, a);


			sweepFinger(showHand, Hand::INDEX, -20, imgBw);
			float refineScale = refineFinger(showHand, Hand::INDEX, imgBw);

			sweepFinger(showHand, Hand::MIDDLE, -15, imgBw);
			refineFinger(showHand, Hand::MIDDLE, imgBw);

			sweepFinger(showHand, Hand::RING, -15, imgBw);
			refineFinger(showHand, Hand::RING, imgBw);

			sweepFinger(showHand, Hand::LITTLE, -15, imgBw);
			refineFinger(showHand, Hand::LITTLE, imgBw);

			sweepThumb1(showHand, imgBw);
			sweepThumb2(showHand, imgBw);

			//showHand.affineFinger(Hand::THUMB1, 1, 20);
			//showHand.affineThumb1(90);
			showHand.showHand(img, cv::Scalar(0.255,255),2);
			//showHand.showMeasureLinePoints(img, cv::Scalar(255,0,0),2);
			//showHand.showControlPoints(img,cv::Scalar(255,0,0),6);
			//showHand.showMeasurePoints(img, cv::Scalar(0,255,0),3);

			//t = (double)getTickCount() - t;
			//printf("detection time = %gms\n", t*1000./cv::getTickFrequency());


			cv::flip(img,img,1);
			cv::imshow("result", img);
			cv::waitKey(10);

			if (countTracking < 7)
			{
				countTracking++;
			} 
			else
			{
				float weight = showHand.calWeight(imgBw);

				if (weight < THRESHOLD)
				{
					countWeight++;
				} 
				else
				{
					countWeight = 0;
				}

				if (countWeight >=3)
				{
					detect = true;
					countWeight = 0;
					countTracking = 0;
					keep = false;
				} 
				else
				{
					cv::Point handCenter;
					handCenter.x = showHand.GetGravity().x;
					handCenter.x = imgWidth - x;
					handCenter.y = showHand.GetGravity().y;

					if (abs(handCenter.x - handCenterX) > 7 || abs(handCenter.y - handCenterY)>5)
					{
						mouse_event(MOUSEEVENTF_MOVE, (handCenter.x - handCenterX) * 6.0, (handCenter.y - handCenterY) * 4.0, 0, 0);
					}
					else
					{
						mouse_event(MOUSEEVENTF_MOVE, (handCenter.x - handCenterX) * 3.0, (handCenter.y - handCenterY) * 2.0, 0, 0);
					}

					handCenterX = handCenter.x;
					handCenterY = handCenter.y;

					//cout<<"weight"<<refineScale<<endl;
					if (refineScale < 0.5)
					{
						mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
					}
					else
					{
						mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
					}
				}
			}

		}
	}

//	while(1)
//	{
//		capture>>img;
//		t = (double)getTickCount();
//		//TSLskinSegment(img, imgBw);
//		skinDetector.skinDetectForUser(img, imgBw);
//
//		
//		//cv::imshow("bw1", imgBw);
//		//medianBlur(imgBw, imgBw, 5);
//		cv::morphologyEx(imgBw, imgBw, MORPH_OPEN, element,cv::Point(-1,-1), 1);
//		cv::morphologyEx(imgBw, imgBw, MORPH_CLOSE, element,cv::Point(-1,-1), 1);
//		cv::findContours(imgBw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//		cv::drawContours(imgBw,contours,-1,255,CV_FILLED,8,hierarchy,2);
//		imshow("bw", imgBw);
//
//		if (!keep)
//		{
//#pragma omp parallel for
//			for (int i = 0; i<NUM_PARTICLES; ++i)
//			{
//				Hand trasitHand(originalHand);
//
//				particles[i] = transition(particles[i], imgWidth, imgHeight, rng);
//
//				x = cvRound(particles[i].x - originalHand.GetGravity().x);
//				y = cvRound(particles[i].y - originalHand.GetGravity().y);
//				s = particles[i].s;
//				a = particles[i].a;
//
//				a = 180*a/CV_PI;
//
//				trasitHand.affineHand(x, y, s, a);
//				particles[i].w = trasitHand.calWeight(imgBw);
//				//particles[i].w = trasitHand.calPalmWeight(imgBw);
//			}
//		}
//
//		Hand preHand(originalHand);
//		x = cvRound(preParitcle.x - originalHand.GetGravity().x);
//		y = cvRound(preParitcle.y - originalHand.GetGravity().y);
//		s = preParitcle.s;
//		a = preParitcle.a;
//
//		a = 180*a/CV_PI;
//
//		preHand.affineHand(x, y, s, a);
//
//		preParitcle.w = preHand.calWeight(imgBw);
//		//preParticles.w = preHand.calPalmWeight(imgBw);
//
//		qsort(particles, NUM_PARTICLES, sizeof(particle), &particle_cmp);
//
//		if (preParitcle.w < particles[0].w)
//		{
//			preParitcle = particles[0];
//
//			normalize_weights(particles, NUM_PARTICLES);
//
//			newParticles = resample(particles, NUM_PARTICLES);
//
//			free(particles);
//			particles = newParticles;
//
//			keep = false;
//		} 
//		else
//		{
//			keep = true;
//		}
//
//		x = cvRound(preParitcle.x - originalHand.GetGravity().x);
//		y = cvRound(preParitcle.y - originalHand.GetGravity().y);
//		s = preParitcle.s;
//		a = preParitcle.a;
//
//		a = 180*a/CV_PI;
//
//		Hand showHand(originalHand);
//		showHand.affineHand(x, y, s, a);
//
//		sweepFinger(showHand, Hand::INDEX, -20, imgBw);
//		refineFinger(showHand, Hand::INDEX, imgBw);
//
//		sweepFinger(showHand, Hand::MIDDLE, -15, imgBw);
//		refineFinger(showHand, Hand::MIDDLE, imgBw);
//
//		sweepFinger(showHand, Hand::RING, -15, imgBw);
//		refineFinger(showHand, Hand::RING, imgBw);
//
//		sweepFinger(showHand, Hand::LITTLE, -15, imgBw);
//		refineFinger(showHand, Hand::LITTLE, imgBw);
//
//		sweepThumb1(showHand, imgBw);
//		sweepThumb2(showHand, imgBw);
//
//		//showHand.affineFinger(Hand::THUMB1, 1, 20);
//		//showHand.affineThumb1(90);
//		showHand.showHand(img, cv::Scalar(0.255,255),2);
//		//showHand.showMeasureLinePoints(img, cv::Scalar(255,0,0),2);
//		//showHand.showControlPoints(img,cv::Scalar(255,0,0),6);
//		//showHand.showMeasurePoints(img, cv::Scalar(0,255,0),3);
//
//		t = (double)getTickCount() - t;
//		printf("detection time = %gms\n", t*1000./cv::getTickFrequency());
//
//
//		cv::flip(img,img,1);
//		cv::imshow("result", img);
//
//		if (waitKey(20)>=0)
//		{
//			break;
//		}	
//	}
	return 0;
}