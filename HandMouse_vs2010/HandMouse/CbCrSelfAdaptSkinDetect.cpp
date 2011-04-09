#include "CbCrSelfAdaptSkinDetect.h"


CbCrSelfAdaptSkinDetect::~CbCrSelfAdaptSkinDetect(void)
{
}

//cr ---Y
//cb ---X

void CbCrSelfAdaptSkinDetect::skinDetectForUser(const Mat& _img, Mat& _mask) const
{
	skinDetect(_img, _mask, up1IntcptBest, up2IntcptBest, down1IntcptBest, down2IntcptBest);
}

void CbCrSelfAdaptSkinDetect::skinDetect(const Mat& _img, Mat& _mask,
	const float&_up1Intcpt,  const float&_up2Intcpt,
	const float&_down1Intcpt, const float&_down2Intcpt) const
{
	float b, g, r;
	vector<cv::Mat> planes;
	cv::split(_img, planes);

	cv::MatIterator_<uchar> it_B = planes[0].begin<uchar>(),
		it_B_end = planes[0].end<uchar>(),
		it_G = planes[1].begin<uchar>(),
		it_R = planes[2].begin<uchar>(),
		it_bw = _mask.begin<uchar>();

	float Cbpix, Crpix;

	for (;it_B != it_B_end; ++it_B,++it_G,++it_R,++it_bw)
	{
		b = *it_B;
		g = *it_G;
		r = *it_R;
		Crpix = -0.0813*b - 0.4187*g + 0.5*r + 128;
		Cbpix = 0.5*b - 0.3313*g - 0.1687*r + 128;

		*it_bw = 255*(Cbpix * UP1SLOPE + _up1Intcpt < Crpix
			&&
			Cbpix * UP2SLOPE + _up2Intcpt < Crpix
			&&
			Cbpix * DOWN1SLOPE + _down1Intcpt > Crpix
			&&
			Cbpix * DOWN2SLOPE + _down2Intcpt > Crpix);
	}
}

const float CbCrSelfAdaptSkinDetect::findFitnessMask(const Mat& _img, const Mat& _mask,
	const float&_up1Intcpt,  const float&_up2Intcpt,
	const float&_down1Intcpt, const float&_down2Intcpt) const 
{

	Mat grayImg;
	grayImg = Mat::zeros(_img.size(), CV_8UC1);
	skinDetect(_img, grayImg, _up1Intcpt, _up2Intcpt, _down1Intcpt, _down2Intcpt);

	int i, j;
	int areaOfMask = 0;
	float skinMaskCount = 0;
	float BGMaskCount = 0;
	uchar*grayImgData = (uchar*)grayImg.data;
	uchar*maskData = (uchar*)_mask.data;
	int totalTmp = _img.cols * _img.rows;
	for (i = 0; i < totalTmp; i++, grayImgData++, maskData++)
	{
		if ((int)grayImgData[0] > 0)
		{
			if ((int)maskData[0] > 0)
			{
				skinMaskCount = skinMaskCount + 1;
				areaOfMask++;
			}
			else
			{
				BGMaskCount = BGMaskCount + 1;
			}
		}
		else if ((int)maskData[0] > 0)
		{
			areaOfMask++;
		}
	}
	float fiter = 2 * skinMaskCount / areaOfMask - 
		BGMaskCount / ((float)(_img.rows * _img.cols - areaOfMask));
	return fiter;

}


const float CbCrSelfAdaptSkinDetect::findFitnessROI(const Mat& _img, const Rect& _ROI,
	const float&_up1Intcpt,  const float&_up2Intcpt, 
	const float&_down1Intcpt, const float&_down2Intcpt) const
{
	Mat grayImg;
	grayImg = Mat::zeros(_img.size(), CV_8UC1);
	skinDetect(_img, grayImg, _up1Intcpt, _up2Intcpt, _down1Intcpt, _down2Intcpt);
	int i, j;

	uchar*grayImgData = (uchar*)grayImg.data;
	float skinMaskCount = 0;
	float BGMaskCount = 0;

	int totalTmp = _img.cols * _img.rows;
	for (i = 0; i < totalTmp; i++, grayImgData++)
	{
		if ((int)grayImgData[0] > 0)
		{
			if (i >= _ROI.y && i < (_ROI.y + _ROI.height) && j >= _ROI.x && (j < _ROI.x + _ROI.width))
			{
				skinMaskCount = skinMaskCount + 1;
			}
			else
			{
				BGMaskCount = BGMaskCount + 1;
			}
		}

	}
	float fiter = 2 * skinMaskCount / ((float)(_ROI.width * _ROI.height)) - 
		BGMaskCount / ((float)(_img.rows * _img.cols - _ROI.width * _ROI.height));
	return fiter;
}


void CbCrSelfAdaptSkinDetect::getBestParamentMask(const Mat& _img, const Mat& _mask)
{
	int i, j;
	float bestPara;
	int bestIdx = -1000;

	float tmpPara;

	for (i = 0; i < 21; i++)
	{
		up1IntcptBest = UP1INTCPT + i -5;

		tmpPara = findFitnessMask(_img, _mask,
			up1IntcptBest, UP2INTCPT, DOWN1INTCPT, DOWN2INTCPT);
		if (bestPara < tmpPara)
		{
			bestPara = tmpPara;
			bestIdx = i;

		}
	}
	up1IntcptBest = UP1INTCPT + bestIdx -5;

	bestPara = -1000;

	for (i = 0; i < 21; i++)
	{
		up2IntcptBest = UP2INTCPT + i -15;

		tmpPara = findFitnessMask(_img, _mask,
			up1IntcptBest, up2IntcptBest, DOWN1INTCPT, DOWN2INTCPT);
		if (bestPara < tmpPara)
		{
			bestPara = tmpPara;
			bestIdx = i;
		}
	}
	up2IntcptBest = UP2INTCPT + bestIdx -15;

	bestPara = -1000;

	for (i = 0; i < 21; i++)
	{
		down1IntcptBest = DOWN1INTCPT + i -15;

		tmpPara = findFitnessMask(_img, _mask,
			up1IntcptBest, up2IntcptBest, down1IntcptBest, DOWN2INTCPT);
		if (bestPara < tmpPara)
		{
			bestPara = tmpPara;
			bestIdx = i;
		}
	}

	down1IntcptBest = DOWN1INTCPT + bestIdx -15;

	bestPara = -1000;

	for (i = 0; i < 21; i++)
	{
		down2IntcptBest = DOWN2INTCPT + i -15;

		tmpPara = findFitnessMask(_img, _mask,
			up1IntcptBest, up2IntcptBest, down1IntcptBest, down2IntcptBest);
		if (bestPara < tmpPara)
		{
			bestPara = tmpPara;
			bestIdx = i;
		}
	}
	down2IntcptBest = DOWN2INTCPT + bestIdx -15;
}

void CbCrSelfAdaptSkinDetect::getBestParamentROI(const Mat& _img, const Rect& _ROI)
{
	int i, j;
	float bestPara;
	int bestIdx = -1000;

	float tmpPara;

	for (i = 0; i < 21; i++)
	{
		up1IntcptBest = UP1INTCPT + i -5;

		tmpPara = findFitnessROI(_img, _ROI,
			up1IntcptBest, UP2INTCPT, DOWN1INTCPT, DOWN2INTCPT);
		if (bestPara < tmpPara)
		{
			bestPara = tmpPara;
			bestIdx = i;

		}
	}
	up1IntcptBest = UP1INTCPT + bestIdx -5;

	bestPara = -1000;

	for (i = 0; i < 21; i++)
	{
		up2IntcptBest = UP2INTCPT + i -15;

		tmpPara = findFitnessROI(_img, _ROI,
			up1IntcptBest, up2IntcptBest, DOWN1INTCPT, DOWN2INTCPT);
		if (bestPara < tmpPara)
		{
			bestPara = tmpPara;
			bestIdx = i;
		}
	}
	up2IntcptBest = UP2INTCPT + bestIdx -15;

	bestPara = -1000;

	for (i = 0; i < 21; i++)
	{
		down1IntcptBest = DOWN1INTCPT + i -15;

		tmpPara = findFitnessROI(_img, _ROI,
			up1IntcptBest, up2IntcptBest, down1IntcptBest, DOWN2INTCPT);
		if (bestPara < tmpPara)
		{
			bestPara = tmpPara;
			bestIdx = i;
		}
	}

	down1IntcptBest = DOWN1INTCPT + bestIdx -15;

	bestPara = -1000;

	for (i = 0; i < 21; i++)
	{
		down2IntcptBest = DOWN2INTCPT + i -15;

		tmpPara = findFitnessROI(_img, _ROI,
			up1IntcptBest, up2IntcptBest, down1IntcptBest, down2IntcptBest);
		if (bestPara < tmpPara)
		{
			bestPara = tmpPara;
			bestIdx = i;
		}
	}
	down2IntcptBest = DOWN2INTCPT + bestIdx -15;
}