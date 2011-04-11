#ifndef TSL_SKIN_SEG_H
#define TSL_SKIN_SEG_H

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <math.h>


const double EPSILON = 1e-20;
const double PI = 3.1415;

void TSLskinSegment(const cv::Mat &src, cv::Mat &dst);

#endif