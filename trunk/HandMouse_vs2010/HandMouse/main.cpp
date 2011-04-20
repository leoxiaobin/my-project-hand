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

const string hogXlm = "hog_64x64_16x32_8x16_8x16_2.xml";

const int NUM_PARTICLES = 1000;

const float THRESHOLD = 1e10;


int main()
{
	std::ofstream out("out.txt");
	//Hand originalHand(confile, cntpntfile, mlinefile);
	//HOGDescriptor HandClassifier("c:\\OpenCV\\HOG\\hog_64x64_16x32_8x16_8x16_2.xml");
	HOGDescriptor HandClassifier(hogXlm);

	Hand originalHand("newhand.xml");

	//originalHand.affineHand(0, -50, 0.6, 0);
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
		medianBlur(imgBw,imgBw,3);
	
		originalHand.showHand(img,Scalar(255,255,0),3);
		//originalHand.showMeasureLinePoints(img, cv::Scalar(255,0,255),2);
		cv::flip(img,img,1);
		imshow("result",img);
		imshow("test",imgBw);
	
		if(originalHand.calWeight(imgBw)>1) 
		{
			Mat _img, _mask;
			cv::resize(img, _img, cv::Size(160, 120));
			cv::resize(mask, _mask, cv::Size(160, 120));
			skinDetector.getBestParamentMask(_img, _mask);
			countFit++;
		}

		if(countFit == 4) break;

		cv::waitKey(20);

	}

	cv::Mat element = cv::Mat::ones(3,3,CV_8UC1);
	vector< vector<cv::Point> > contours;
	vector< Vec4i> hierarchy;

	int countDetectHand(0), countTracking(0), countWeight(0);

	bool detect = true;
	bool keep = false;
	int click = 0;
	int deleteClick = 0;
	int witchKeyBoard = 0;
	bool firstGetMouse = true;
	POINT mousePosition;

	int littleClick= 0;
	int indexClick = 0;
	int middleClick = 0;
	int ringClick = 0;

	int handCenterX(0), handCenterY(0);
	Hand hand(originalHand);
	Hand preHand(hand);

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
				//rectangle(img, foundHands[0].tl(), foundHands[0].br(),cv::Scalar(0,0,233),3);
			} 
			else
			{
				countDetectHand = 0;
			}

			
			if (countDetectHand == 3)
			{
				hand = originalHand;
				handROI = foundHands[0];

				countDetectHand = 0;

				detect = false;

				cv::Point2f moveHand;
				moveHand.x = handROI.x - originalHand.GetGravity().x + 0.5*handROI.width;
				moveHand.y = handROI.y - originalHand.GetGravity().y + 0.5*handROI.height;

				float handScale = 1;

				//rectangle(img, handROI.tl(), handROI.br(), cv::Scalar(0,233,2),3);
				//cout<<"hand roi"<<handROI.area()<<endl;
				//cout<<"hand contour"<<originalHand.GetArea()<<endl;
				handScale = 1.8*(float)handROI.area()/(float)originalHand.GetArea();
				//cout<<"scale: "<<handScale<<endl;

				//hand = originalHand;

				hand.affineHand(moveHand.x, moveHand.y, handScale, 0);
				preParitcle.w = 0;
				preParitcle.x = hand.GetGravity().x;
				preParitcle.y = hand.GetGravity().y;
				preParitcle.s = 1;
				preParitcle.a = 0;

				particles = init_distribution(hand.GetGravity(), NUM_PARTICLES);
				preHand = hand;

			}
			foundHands.clear();

			cv::waitKey(10);
		} 
		else
		{
			t = (double)cvGetTickCount();
			//TSLskinSegment(img, imgBw);
			skinDetector.skinDetectForUser(img, imgBw);
			//cv::medianBlur(imgBw, imgBw, 3);
			cv::morphologyEx(imgBw, imgBw, MORPH_OPEN, element,cv::Point(-1,-1), 1);
			cv::morphologyEx(imgBw, imgBw, MORPH_CLOSE, element,cv::Point(-1,-1), 1);
			cv::findContours(imgBw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			cv::drawContours(imgBw,contours,-1,255,CV_FILLED,8,hierarchy,1);

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
			else
			{
				cout<<"keep!"<<endl;
			}

			//Hand preHand(hand);
			//x = cvRound(preParitcle.x - originalHand.GetGravity().x);
			//y = cvRound(preParitcle.y - originalHand.GetGravity().y);
			//s = preParitcle.s;
			//a = preParitcle.a;

			//a = 180*a/CV_PI;

			//preHand.affineHand(x, y, s, a);

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
			float indexScale = refineFinger(showHand, Hand::INDEX, imgBw);

			sweepFinger(showHand, Hand::MIDDLE, -15, imgBw);
			float middleScale = refineFinger(showHand, Hand::MIDDLE, imgBw);

			sweepFinger(showHand, Hand::RING, -15, imgBw);
			float ringScale = refineFinger(showHand, Hand::RING, imgBw);

			sweepFinger(showHand, Hand::LITTLE, -15, imgBw);
			float littleScale = refineFinger(showHand, Hand::LITTLE, imgBw);

			sweepThumb1(showHand, imgBw);
			sweepThumb2(showHand, imgBw);

			preHand.showHand(img, cv::Scalar(233,1,1),2);
			
			cv::line(img, showHand.GetGravity(), preHand.GetGravity(), cv::Scalar(0,0,233),3);

			float dx, dy;
			dx = preHand.GetGravity().x - showHand.GetGravity().x;
			dy = showHand.GetGravity().y - preHand.GetGravity().y;

			preHand = showHand;

			showHand.showHand(img, cv::Scalar(0.255,255),2);

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
					firstGetMouse = true;
					//SetCursorPos(1280, 720);
					//handCenterX = imgWidth - showHand.GetGravity().x;
					//handCenterY = showHand.GetGravity().y;
				} 
				else
				{
					cv::Point handCenter;
					handCenter.x = showHand.GetGravity().x;
					handCenter.x = imgWidth - handCenter.x;
					handCenter.y = showHand.GetGravity().y;
					//out<<showHand.GetGravity().x<<"  "<<showHand.GetGravity().y<<endl;
					/*if (firstGetMouse)
					{
					handCenterX = handCenter.x;
					handCenterY = handCenter.y;
					firstGetMouse = false;
					}*/

					//if (abs(handCenter.x - handCenterX) > 10 || abs(handCenter.y - handCenterY)>8)
					//{
					//	mouse_event(MOUSEEVENTF_MOVE, (handCenter.x - handCenterX) * 8.0, (handCenter.y - handCenterY) * 6.0, 0, 0);
					//}
					//else if(abs(handCenter.x - handCenterX) > 8 || abs(handCenter.y - handCenterY)>6)
					//{
					//	mouse_event(MOUSEEVENTF_MOVE, (handCenter.x - handCenterX) * 2.0, (handCenter.y - handCenterY) * 2.0, 0, 0);
					//}
					//else if(abs(handCenter.x - handCenterX) > 4 || abs(handCenter.y - handCenterY)>2)
					//{
					//	mouse_event(MOUSEEVENTF_MOVE, (handCenter.x - handCenterX) * 1.0, (handCenter.y - handCenterY) * 1.0, 0, 0);
					//}
					//else
					//{
					//	mouse_event(MOUSEEVENTF_MOVE, 0, 0, 0, 0);
					//}

					if (abs(dx) > 10 || abs(dy)>8)
					{
						mouse_event(MOUSEEVENTF_MOVE, (dx) * 8.0, (dy) * 6.0, 0, 0);
					}
					else if(abs(dx) > 8 || abs(dy)>6)
					{
						mouse_event(MOUSEEVENTF_MOVE, (dx) * 2.0, (dy) * 2.0, 0, 0);
					}
					//else if(abs(dx) > 4 || abs(dy)>2)
					//{
					//	mouse_event(MOUSEEVENTF_MOVE, (dx) * 1.5, (dy) * 1.5, 0, 0);
					//}
					//else
					//{
					//	mouse_event(MOUSEEVENTF_MOVE, 0, 0, 0, 0);
					//}

					GetCursorPos(&mousePosition);
					handCenterX = handCenter.x;
					handCenterY = handCenter.y;

					//cout<<"weight"<<refineScale<<endl;
					if (indexScale<0.5)
					{
						indexClick++;
					}
					if (middleScale<0.5)
					{
						middleClick++;
					}
					if (ringScale<0.5)
					{
						ringClick++;
					}
					if (littleScale<0.5)
					{
						littleClick++;
					}


					//if (indexClick >=2 && indexScale>0.5)
					//{
					//	mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
					//	mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
					//	indexClick = 0;
					//}

					if (indexClick >= 2 && indexScale>0.5)
					{
						mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
						mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
						indexClick = 0;
					}
					else if (middleClick >= 2 && middleScale>0.5)
					{
						keybd_event(VK_DOWN , 0xE051, KEYEVENTF_EXTENDEDKEY | 0, 0);
						keybd_event(VK_DOWN , 0xE051, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
						middleClick = 0;
					}
					else if (ringClick >= 2 && ringScale>0.5)
					{
						keybd_event(VK_UP , 0xE048, KEYEVENTF_EXTENDEDKEY | 0, 0);
						keybd_event(VK_UP , 0xE048, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
						ringClick = 0;
					}
					//else if (littleClick >= 2 && littleScale>0.5)
					//{
					//	keybd_event(VK_ESCAPE  , 0x1, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//	keybd_event(VK_ESCAPE  , 0x1, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//	ringClick = 0;
					//}


					//if (littleClick == 2)
					//{
					//	witchKeyBoard++;
					//	cout<<witchKeyBoard<<endl;
					//	witchKeyBoard %= 3;
					//}

					//if (indexClick >= 2 && indexScale>0.5) 
					//{
					//	keybd_event(VK_DOWN , 0xE051, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//	keybd_event(VK_DOWN , 0xE051, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//	indexClick = 0;
					//}
					//else if (middleClick >= 2 && middleScale>0.5)
					//{
					//	keybd_event(VK_UP  , 0xE048, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//	keybd_event(VK_UP  , 0xE048, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//	cout<<"click"<<endl;
					//	middleClick = 0;
					//}


					//else if (ringClick == 2 && ringScale>0.5)
					//{
					//	keybd_event(0x33 , 0x4, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//	keybd_event(0x33 , 0x4, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//	ringClick = 0;
					//}

					//switch (witchKeyBoard)
					//{
					//case 0:
					//	if (indexClick >= 2 && indexScale>0.5)
					//	{
					//		keybd_event(0x31 , 0x2, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x31 , 0x2, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		indexClick = 0;
					//	}
					//	if (middleClick == 2 && middleScale>0.5)
					//	{
					//		keybd_event(0x32 , 0x3, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x32 , 0x3, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		middleClick = 0;
					//	}
					//	if (ringClick == 2 && ringScale>0.5)
					//	{
					//		keybd_event(0x33 , 0x4, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x33 , 0x4, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		ringClick = 0;
					//	}
					//	break;
					//case 1:
					//	if (indexClick >= 2 && indexScale>0.5)
					//	{
					//		keybd_event(0x34 , 0x5, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x34 , 0x5, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		indexClick = 0;
					//	}
					//	if (middleClick == 2 && middleScale>0.5)
					//	{
					//		keybd_event(0x35 , 0x6, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x35 , 0x6, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		middleClick = 0;
					//	}
					//	if (ringClick == 2 && ringScale>0.5)
					//	{
					//		keybd_event(0x36 , 0x7, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x36 , 0x7, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		ringClick = 0;
					//	}
					//	break;
					//case 2:
					//	if (indexClick >= 2 && indexScale>0.5)
					//	{
					//		keybd_event(0x37 , 0x8, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x37 , 0x8, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		indexClick = 0;
					//	}
					//	if (middleClick == 2 && middleScale>0.5)
					//	{
					//		keybd_event(0x38 , 0x9, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x38 , 0x9, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		middleClick = 0;
					//	}
					//	if (ringClick == 2 && ringScale>0.5)
					//	{
					//		keybd_event(0x39 , 0x0A, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x39 , 0x0A, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		ringClick = 0;
					//	}
					//	witchKeyBoard = -1;
					//	break;
					//case -1:
					//	if (indexClick >= 2 && indexScale>0.5)
					//	{
					//		keybd_event(0x37 , 0x8, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x37 , 0x8, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		indexClick = 0;
					//	}
					//	if (middleClick == 2 && middleScale>0.5)
					//	{
					//		keybd_event(0x38 , 0x9, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x38 , 0x9, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		middleClick = 0;
					//	}
					//	if (ringClick == 2 && ringScale>0.5)
					//	{
					//		keybd_event(0x39 , 0x0A, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//		keybd_event(0x39 , 0x0A, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//		ringClick = 0;
					//	}
					//	witchKeyBoard = -1;
					//	break;
					//}

					//if (littleScale < 0.5)
					//{
					//	deleteClick ++;
					//}
					//if (deleteClick == 2)
					//{
					//	keybd_event(VK_BACK , 0x0E, KEYEVENTF_EXTENDEDKEY | 0, 0);
					//	keybd_event(VK_BACK , 0x0E, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					//	deleteClick = 0;
					//	//cout<<"click!!"<<endl;
					//}

					

				}
			}
		}
		t = (double)cvGetTickCount() - t;
		printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
		cv::flip(img,img,1);
		cv::imshow("result", img);
		cv::waitKey(10);
	}
	return 0;
}