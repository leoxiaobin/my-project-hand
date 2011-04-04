#ifndef TSL_SKIN_SEG_H
#define TSL_SKIN_SEG_H

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <math.h>


const double EPSILON = 0.0000001;
const double PI = 3.1415;

void TSLskinSegment(const cv::Mat &src, cv::Mat &dst);

#endif