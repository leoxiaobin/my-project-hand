#include "TSLskinSeg.h"

void TSLskinSegment(const cv::Mat &src, cv::Mat& dst)
{
	double r, g, b, T, S, rgbSum;
	vector<cv::Mat> planes;
	cv::split(src,planes);

	cv::MatIterator_<uchar> it_B = planes[0].begin<uchar>(),
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
			T = (atan(r/g)/(2*PI)+0.25)*300;
		} 
		else
		{
			T = ((atan(r/g))/(2*PI)+0.75)*300;
		}

		S = (sqrt((r*r+g*g)*1.8))*100;

		*it_bw = 255*(T>125&&T<185&&((1.033*T-114.8425)>S)&&((380.1575-1.967*T)>S));
	}
}