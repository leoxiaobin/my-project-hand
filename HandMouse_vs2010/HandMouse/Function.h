#ifndef FUNCTION_H
#define FUNCTION_h
#include "Hand.h"

void sweepFinger(Hand &hand, Hand::FINGER finger,float initialAngle, const cv::Mat& bw);
void refineFinger(Hand &hand, Hand::FINGER finger, const cv::Mat& bw);
void sweepThumb1(Hand &hand, const cv::Mat& bw);
void sweepThumb2(Hand &hand, const cv::Mat& bw);

#endif