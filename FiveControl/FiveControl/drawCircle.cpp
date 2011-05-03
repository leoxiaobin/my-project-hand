#include "drawCircle.h"
#include <cmath>

drawCircle::drawCircle(int _num_divided, int _out_radius, int _in_radius, cv::Point _center):
	num_divided(_num_divided), out_radius(_out_radius), in_radius(_in_radius), center(_center), draw_index(1)
{
	//line_points.clear();
	
	cv::Point temp_point(0,0);
	for (int i = 0; i<_num_divided; ++i)
	{
		std::vector<cv::Point> per_points(100, temp_point);
		line_points.push_back(per_points);
		//per_points.reserve(100);
		//for (int j = 0; j<50; ++j)
		//{
		//	temp_point.x = out_radius*sin(2*CV_PI*i/_num_divided + angle_step*j)
		//}
	}
	reset(_out_radius, _in_radius, _center);
};

drawCircle::~drawCircle()
{

};

void drawCircle::reset(int _out_radius, int _in_radius, cv::Point _center)
{
	center = _center;
	float angle_step = CV_PI*0.04/num_divided;
	for (int i = 0; i<num_divided; ++i)
	{
		for (int j = 0; j<50; ++j)
		{
			line_points[i][j].x = out_radius*sin(2*CV_PI*i/num_divided + angle_step*j)+_center.x;
			line_points[i][j].y = out_radius*cos(2*CV_PI*i/num_divided + angle_step*j)+_center.y;
			line_points[i][99-j].x = in_radius*sin(2*CV_PI*i/num_divided + angle_step*j)+_center.x;
			line_points[i][99-j].y = in_radius*cos(2*CV_PI*i/num_divided + angle_step*j)+_center.y;		
		}
	}
}

void drawCircle::drawRing(cv::Mat &img, cv::Scalar color)
{
	for (int i=0; i<draw_index; ++i)
	{
		cv::fillConvexPoly(img, &line_points[i][0], 100, color);
	}
	if (draw_index == num_divided)
	{
		draw_index = 1;
	} 
	else
	{
		++draw_index;
	}
}