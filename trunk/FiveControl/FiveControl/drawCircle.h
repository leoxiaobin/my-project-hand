#ifndef DRAW_CIRLCE_H
#define DRAW_CIRCLE_H

#include <cv.h>
#include <highgui.h>
#include <vector>

class drawCircle
{
public:
	//drawCircle();
	drawCircle(int _num_divided, int _out_radius, int _in_radius, cv::Point _center);
	~drawCircle();
	int draw_index;
	void drawRing(cv::Mat &img, cv::Scalar color);
	void reset(int _out_radius, int _in_radius, cv::Point _center);
private:
	int num_divided;
	int out_radius;
	int in_radius;
	int num_points;
	cv::Point center;
	std::vector< std::vector<cv::Point> > line_points; 
};
#endif