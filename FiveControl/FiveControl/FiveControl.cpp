#include "FiveControl.h"
const int FiveControl::angle[rotateNum] = {0, -15, 15};

FiveControl::FiveControl(const string five, const string fist)
{
	load(five, fist);
}

FiveControl::FiveControl()
{

}

FiveControl::~FiveControl()
{

}

bool FiveControl::load(const string five, const string fist)
{
	if (fistClassifier.load(fist) && fiveClassifer.load(five))
	{
		return true;
	} 
	else
	{
		return false;
	}
}

void FiveControl::TSLskinSegment(const cv::Mat& src, cv::Mat& dst)
{
	double r, g, b, T, S, rgbSum;
	vector<Mat> planes;
	split(src,planes);

	MatIterator_<uchar> it_B = planes[0].begin<uchar>(),
		it_B_end = planes[0].end<uchar>(),
		it_G = planes[1].begin<uchar>(),
		it_R = planes[2].begin<uchar>(),
		it_bw = dst.begin<uchar>();

	for (;it_B != it_B_end; ++it_B,++it_G,++it_R,++it_bw)
	{
		b = *it_B;
		g = *it_G;
		r = *it_R;
		rgbSum = b+g+r;
		b = b/rgbSum-0.33;
		r = r/rgbSum -0.33;
		g = g/rgbSum-0.33;

		if (abs(g)<EPSILON)
		{
			T = 0;
		}
		else if (g>EPSILON)
		{
			T = (atan(r/g)/(2*CV_PI)+0.25)*300;
		} 
		else
		{
			T = ((atan(r/g))/(2*CV_PI)+0.75)*300;
		}

		S = (sqrt((r*r+g*g)*1.8))*100;

		*it_bw = 255*(T>125&&T<185&&((1.033*T-114.8425)>S)&&((380.1575-1.967*T)>S));
	}
}

void FiveControl::detectFive(const cv::Mat& _img, cv::Rect& ROIRect, vector<cv::Rect>& five)
{
	five.clear();
	cv::Mat img = _img(ROIRect);
	if (_img.rows == ROIRect.height)
	{
		fiveClassifer.detectMultiScale(img, five, 1, Size(8,8), Size(0,0), 1.2, 3);
	} 
	else
	{
		cv::Mat rotatedImg;
		img.copyTo(rotatedImg);

		cv::Mat rotMat(2, 3, CV_32FC1);
		cv::Point center(img.cols/2, img.rows/2);
		bool findFive = false;
		for (int i = 0; i < rotateNum; i++)
		{
			rotMat = cv::getRotationMatrix2D(center, angle[i], 1);
			cv::warpAffine(img, rotatedImg, rotMat, img.size());
			fiveClassifer.detectMultiScale(rotatedImg, five, 1, Size(8,8), Size(0,0), 1.2, 3);
			if (five.size() != 0)
			{
				findFive = true;
			}
		}
	}
}

void FiveControl::detectFist(const cv::Mat& _img, cv::Rect& ROIRect, vector<cv::Rect>& fist)
{
	fist.clear();
	Mat img = _img(ROIRect);
	if (_img.rows == ROIRect.height)
	{
		fistClassifier.detectMultiScale(img, fist, 0.2, 1.1, 3, 0, Size(24,24));
	} 
	else
	{
		Mat rotatedImg;
		img.copyTo(rotatedImg);

		Mat rotMat(2, 3, CV_32FC1);
		Point center(img.cols/2, img.rows/2);
		bool findFist = false;
		for (int i = 0; i < rotateNum && !findFist; i++)
		{
			rotMat = getRotationMatrix2D(center, angle[i], 1);
			warpAffine(img, rotatedImg, rotMat, img.size());
			fistClassifier.detectMultiScale(img, fist, 0.2, 1.1, 3, 0, Size(24,24));
			if (fist.size() != 0)
			{
				findFist = true;
			}
		}
	}
}

void FiveControl::process(cv::Mat& img, int& hand, cv::Rect& handLocation)
{
	int imgHeight, imgWidth;
	imgHeight = img.rows;
	imgWidth = img.cols;

	cv::Mat imgCopy = cv::Mat(imgHeight, imgWidth, img.type());
	cv::Mat imgBw = cv::Mat(imgHeight, imgWidth, CV_8UC1);

	vector<cv::Rect> hands;

	TSLskinSegment(img, imgBw);
	cv::medianBlur(imgBw, imgBw, 5);
	img.copyTo(imgCopy, imgBw);

	cv::Rect handROI;
	if (handLocation.width == imgWidth)
	{
		handROI = handLocation;
	} 
	else
	{
		handROI.x = max(0, handLocation.x - handLocation.width/2);
		handROI.y = max(0, handLocation.y - handLocation.height/2);
		handROI.width = (handROI.x + handLocation.width*2) > imgWidth ? (imgWidth - handROI.x) : handLocation.width*2;
		handROI.height = (handROI.y + handLocation.height*2) > imgHeight ? (imgHeight-handROI.y) : handLocation.height*2;
	}
	detectFive(imgCopy, handROI, hands);

	if (!hands.empty())
	{
		handLocation.x = handROI.x + hands[0].x;
		handLocation.y = handROI.y + hands[0].y;
		handLocation.width = hands[0].width;
		handLocation.height = hands[0].height;
		hand = 1;
	} 
	else
	{
		detectFist(imgCopy, handROI, hands);
		if (!hands.empty())
		{
			handLocation.x = handROI.x + hands[0].x;
			handLocation.y = handROI.y + hands[0].y;
			handLocation.width = hands[0].width;
			handLocation.height = hands[0].height;
			hand = 2;
		} 
		else
		{
			handLocation.x = 0;
			handLocation.y = 0;
			handLocation.width = imgWidth;
			handLocation.height = imgHeight;
			hand = 0;
		}
	}
}