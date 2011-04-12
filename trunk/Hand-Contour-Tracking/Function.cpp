#include "Function.h"

void sweepFinger(Hand &hand, Hand::FINGER finger,float initialAngle, const cv::Mat& bw)
{
	hand.affineFinger(finger, 1, initialAngle);
	AngleWeight angleWeight;
	angleWeight.angle = initialAngle;
	angleWeight.weight = hand.calFingerAngleWeight(bw, finger);

	float angleRange = -2.0*initialAngle;
	float weight(0);
	for (float i = 1; i<angleRange; i++)
	{
		hand.affineFinger(finger, 1, 1);
		weight = hand.calFingerAngleWeight(bw, finger);

		if (weight > angleWeight.weight)
		{
			angleWeight.angle = initialAngle +i;
			angleWeight.weight = weight;
		}
	}
	hand.affineFinger(finger, 1, angleWeight.angle+initialAngle);
}

float refineFinger(Hand &hand, Hand::FINGER finger, const cv::Mat& bw)
{
	ScaleWeight scaleWeight;
	scaleWeight.scale = 1;
	scaleWeight.weight = hand.calFingerScaleWeight(bw, finger);

	float preScale(1);
	float weight(0);

	for (float i = 0.1; i<1.3; i +=0.1)
	{
		hand.affineFinger(finger, 1/preScale*i, 0);
		preScale = i;

		weight = hand.calFingerScaleWeight(bw, finger);

		if (weight >= scaleWeight.weight)
		{
			scaleWeight.scale = i;
			scaleWeight.weight = weight;
		}
	}
	//std::cout<<"ok~!!!!"<<std::endl;
	hand.affineFinger(finger, scaleWeight.scale/preScale, 0);

	return scaleWeight.scale;
	
}

void sweepThumb1(Hand &hand, const cv::Mat& bw)
{
	hand.affineThumb1(-5);
	AngleWeight angleWeight;
	angleWeight.angle = -5;
	angleWeight.weight = hand.calThumb1Weight(bw);

	float angleRange = 50;
	float weight(0);
	for (float i = 1; i<angleRange; i++)
	{
		hand.affineThumb1(1);
		weight = hand.calThumb1Weight(bw);

		if (weight > angleWeight.weight)
		{
			angleWeight.angle = -5 +i;
			angleWeight.weight = weight;
		}
	}
	hand.affineThumb1(angleWeight.angle-45);
}

void sweepThumb2(Hand &hand, const cv::Mat& bw)
{
	hand.affineThumb2(-5);
	AngleWeight angleWeight;
	angleWeight.angle = -5;
	angleWeight.weight = hand.calThumb2Weight(bw);

	float angleRange = 50;
	float weight(0);
	for (float i = 1; i<angleRange; i++)
	{
		hand.affineThumb2(1);
		weight = hand.calThumb2Weight(bw);

		if (weight > angleWeight.weight)
		{
			angleWeight.angle = -5 +i;
			angleWeight.weight = weight;
		}
	}
	hand.affineThumb2(angleWeight.angle-45);
}

void setMask(Hand &hand, cv::Mat& mask)
{
	vector< vector<Point> > contours;
	vector<cv::Point> controlPoints;

	for (int i = 0; i < hand.GetNumBspine(); ++i)
	{
		int numControlPoints = hand.ControlPoints[i].rows;
		for ( int j = 0; j < numControlPoints; ++j)
		{
			cv::Point tmpPoint;
			tmpPoint.x = *hand.ControlPoints[i].ptr<float>(j);
			tmpPoint.y = *(hand.ControlPoints[i].ptr<float>(j)+1);
			controlPoints.push_back(tmpPoint);
		}
	}
	contours.push_back(controlPoints);

	cv::drawContours(mask, contours, -1, cv::Scalar(255,255,255), CV_FILLED);
}