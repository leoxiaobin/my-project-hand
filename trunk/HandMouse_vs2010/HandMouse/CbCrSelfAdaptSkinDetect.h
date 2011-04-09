#pragma once
#include <highgui.h>
#include <cv.h>
#include <cvaux.h>
#include <cxcore.h>
#include <ml.h>
#include <iostream>

using namespace cv;
using namespace std;

const static float	 UP1SLOPE = -1.222222222;
const static float  UP1INTCPT = 267.33333333;

const static float  UP2SLOPE = 0.875;
const static float  UP2INTCPT = 29.375;

const static float  DOWN1SLOPE = -1.333333333;
const static float  DOWN1INTCPT = 316.33333333;

const static float  DOWN2SLOPE = -0.06451612903;
const static float  DOWN2INTCPT = 170.612903225;

class CbCrSelfAdaptSkinDetect
{
public:
	CbCrSelfAdaptSkinDetect(void):up1IntcptBest(UP1INTCPT),
	up2IntcptBest(UP2INTCPT),
	down1IntcptBest(DOWN1INTCPT),
	down2IntcptBest(DOWN2INTCPT){}
	~CbCrSelfAdaptSkinDetect(void);

	void skinDetectForUser(const Mat& _img, Mat& _mask) const;

	void getBestParamentMask(const Mat& _img, const Mat& _mask);

	void getBestParamentROI(const Mat& _img, const Rect& _ROI);


private:
	const float findFitnessMask(const Mat& _img, const Mat& _mask,
		const float&_up1Intcpt,  const float&_up2Intcpt,
		const float&_down1Intcpt, const float&_down2Intcpt) const ;

	const float findFitnessROI(const Mat& _img, const Rect& _ROI,
		const float&_up1Intcpt,  const float&_up2Intcpt, 
		const float&_down1Intcpt, const float&_down2Intcpt) const;

	void skinDetect(const Mat& _img, Mat& _mask,
		const float&_up1Intcpt,  const float&_up2Intcpt,
		const float&_down1Intcpt, const float&_down2Intcpt) const;

	float up1IntcptBest;
	float up2IntcptBest;
	float down1IntcptBest;
	float down2IntcptBest;

};

