#ifndef FIVE_CONTROL_H
#define FIVE_CONTROL_H
#include "opencv.h"

const double EPSILON = 0.0000001;

class FiveControl
{
public:

	FiveControl(const string five, const string fist);
	FiveControl();
	~FiveControl();

	int hand;
	cv::Rect handLocation;

	bool load(const string five, const string fist);
	//void process(cv::Mat& img, int& hand, cv::Rect& handLocation);
	void process(cv::Mat& img);
	//bool ifLoaded;

private:

	static const int rotateNum = 3;
	static const int angle[rotateNum];
	cv::HOGDescriptor fiveClassifer;
	cv::CascadeClassifier fistClassifier;

	void TSLskinSegment(const cv::Mat &src, cv::Mat &dst);

	void detectFive(const cv::Mat& _img, cv::Rect& ROIRect, vector<cv::Rect>& five);
	void detectFist(const cv::Mat& _img, cv::Rect& ROIRect, vector<cv::Rect>& fist);
};
#endif