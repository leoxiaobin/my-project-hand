#ifndef HAND_H
#define HAND_H
#include "opencv.h"
#include "Bspline.h"


class Hand
{
public:
	//Hand();
	Hand(const string confile, const string ctrpntfile, const string mlinefile);
	Hand(const Hand& hand);
	Hand& operator=(const Hand &rhs);
	~Hand();

	enum FINGER { LITTLE = 1, RING = 2, MIDDLE = 3, INDEX = 4, THUMB1 = 5, THUMB2 = 6 } ;

	bool loadTXT(const string confile, const string ctrpntfile, const string mlinefile);
	
	//float a;
	//float s;

	//float ap;
	//float sp;

	//float w;

	//public functions
	inline int GetArea() const {return handArea;}
	inline cv::Point2f GetGravity()const {return Gravity;}
	inline int GetNumBspine()const {return nbsplines;}
	void showHand(cv::Mat& img, const cv::Scalar color, const int thinckness);
	void showControlPoints(cv::Mat& img, const cv::Scalar color, const int thinckness);
	void showMeasurePoints(cv::Mat& img, const cv::Scalar color, const int thinckness);
	void showMeasureLinePoints(cv::Mat& img, const cv::Scalar color, const int thinckness);
	void affineHand(int x, int y, float scale, float angle);
	void affineFinger(Hand::FINGER finger, float scale, float angle);
	float calWeight(cv::Mat& img);

	vector<cv::Mat> MeasurePoints;
	vector<cv::Mat> ControlPoints;
	vector< vector<cv::Mat> > MeasureLinePoints;
	;
private:
	//Hand
	float handArea;
	cv::Point2f Gravity;
	//cv::Point2f PreGravity;
	cv::Mat PivotPoints;
	cv::Mat TipPoints;

	//Bspline
	
	const static int nbsplines = 10;
	const static int order = 3;
	const static float interval;
	const static int nknots[nbsplines];
	const static int nspans[nbsplines];
	const static int knotCounts[nbsplines][8]; 
	
	//private functions
	void showPoint(cv::Mat& img, const vector<cv::Mat> sp, const Scalar& color, const int thickness);
	void calMeasureSlope();
	void calMeasureSlope(vector< vector<float> > &Measureslope);
	void calLinePoints(cv::Mat &line, int x, int y, float k, int begin, int end, bool d, bool d2);
	void calAllMeasureLinePoints();
	
};
#endif

