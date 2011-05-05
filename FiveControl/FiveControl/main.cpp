#include "FiveControl.h"
#include "drawCircle.h"
#include <ctime>
#include <cmath>
#include <Windows.h>
#include <string>
#include <strstream>

const string fiveXml = "hog_64x64_16x32_8x16_8x16_1.xml";
const string fistXml = "c1_lbp_958_18_0.45.xml";
int main()
{
	cv::Mat img;
	cv::VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cout<<"Can not open the camera!"<<endl;
		return -1;
	}

	cap>>img;
	if (img.data == NULL)
	{
		cout<<"Can not read the image from the camera!"<<endl;
		return -1;
	}

	FiveControl control;
	if (!control.load(fiveXml, fistXml))
	{
		cout<<"Load xml file failed!"<<endl;
		return -1;
	}

	clock_t beginTime, endTime;
	int num_segment = 10;
	drawCircle test(num_segment, 50, 25, cv::Point(320, 240));
	int left_count = 0;
	int right_count = 0;
	int up_count = 0;
	int down_count = 0;
	int fist_count = 0;
	int first_detected_five_count = 0;
	vector<cv::Point> centers;
	cv::Point center;
	int lost_count = 0;
	int radius = 0;
	cv::Point triangle_points[3];

	while(1)
	{
		cap>>img;
		beginTime = clock();
		control.process(img);

		if (control.hand == 1)
		{
			cv::rectangle(img, control.handLocation, cv::Scalar(0,233,0));
			lost_count = 0;
			++first_detected_five_count;
			if (first_detected_five_count == 5)
			{
				//--first_detected_five_count;
				center.x = control.handLocation.x + control.handLocation.width/2;
				center.y = control.handLocation.y + control.handLocation.height/2;
				test.reset(control.handLocation.width, control.handLocation.width/2, center);
				radius = control.handLocation.width/2;
				//cv::circle(img, center,control.handLocation.width, cv::Scalar(0,233,2),3);
			}
			if (first_detected_five_count>10)
			{
				cv::circle(img, center,radius, cv::Scalar(0,233,2),3);
				int x = control.handLocation.x + control.handLocation.width/2;
				int y = control.handLocation.y + control.handLocation.height/2;
				int distanceX = x - center.x;
				int distanceY = y - center.y;
				float distance = (float)sqrt((float)(distanceX*distanceX+
					distanceY*distanceY));

				if (distance>control.handLocation.width*2/3)
				{
					test.draw_index = 1;
					if (distanceX>=0 && distanceY>=0)
					{
						if (distanceX>=distanceY)
						{
							++left_count;
							down_count = 0;
							right_count = 0;
							up_count = 0;
							//strstream ss;
							//string s;
							//ss<<(10-left_count)/3;
							//ss>>s;
							//if ((10-left_count)%3 == 0)
							//{
							//	cv::putText(img, s, cv::Point(300,260), 1, 5,cv::Scalar(255,0,0), 4);
							//}
							triangle_points[0].x = center.x + radius*1.25;
							triangle_points[0].y = center.y;
							triangle_points[1].x = triangle_points[2].x = center.x + radius;
							triangle_points[1].y = center.y + radius;
							triangle_points[2].y = center.y - radius;
							cv::line(img, triangle_points[0], triangle_points[1], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[0], triangle_points[2], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[2], triangle_points[1], cv::Scalar(0, 233, 233), 3);
						} 
						else
						{
							++down_count;
							up_count = 0;
							left_count = 0;
							right_count = 0;
							triangle_points[0].y = center.y + radius*1.25;
							triangle_points[0].x = center.x;
							triangle_points[1].y = triangle_points[2].y = center.y + radius;
							triangle_points[1].x = center.x + radius;
							triangle_points[2].x = center.x - radius;
							cv::line(img, triangle_points[0], triangle_points[1], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[0], triangle_points[2], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[2], triangle_points[1], cv::Scalar(0, 233, 233), 3);
						}
					} 
					else if (distanceX<0 && distanceY>=0)
					{
						if (-distanceX>=distanceY)
						{
							++right_count;
							left_count = 0;
							up_count = 0;
							down_count = 0;
							triangle_points[0].x = center.x - radius*1.25;
							triangle_points[0].y = center.y;
							triangle_points[1].x = triangle_points[2].x = center.x - radius;
							triangle_points[1].y = center.y + radius;
							triangle_points[2].y = center.y - radius;
							cv::line(img, triangle_points[0], triangle_points[1], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[0], triangle_points[2], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[2], triangle_points[1], cv::Scalar(0, 233, 233), 3);
						} 
						else
						{
							++down_count;
							up_count = 0;
							left_count = 0;
							right_count = 0;
							triangle_points[0].y = center.y + radius*1.25;
							triangle_points[0].x = center.x;
							triangle_points[1].y = triangle_points[2].y = center.y + radius;
							triangle_points[1].x = center.x + radius;
							triangle_points[2].x = center.x - radius;
							cv::line(img, triangle_points[0], triangle_points[1], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[0], triangle_points[2], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[2], triangle_points[1], cv::Scalar(0, 233, 233), 3);
						}
					} 
					else if (distanceX<0 && distanceY<0)
					{
						if (distanceX<=distanceY)
						{
							++right_count;
							left_count = 0;
							up_count = 0;
							down_count = 0;
							triangle_points[0].x = center.x - radius*1.25;
							triangle_points[0].y = center.y;
							triangle_points[1].x = triangle_points[2].x = center.x - radius;
							triangle_points[1].y = center.y + radius;
							triangle_points[2].y = center.y - radius;
							cv::line(img, triangle_points[0], triangle_points[1], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[0], triangle_points[2], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[2], triangle_points[1], cv::Scalar(0, 233, 233), 3);
						} 
						else
						{
							++up_count;
							down_count = 0;
							left_count = 0;
							right_count = 0;
							triangle_points[0].y = center.y - radius*1.25;
							triangle_points[0].x = center.x;
							triangle_points[1].y = triangle_points[2].y = center.y - radius;
							triangle_points[1].x = center.x + radius;
							triangle_points[2].x = center.x - radius;
							cv::line(img, triangle_points[0], triangle_points[1], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[0], triangle_points[2], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[2], triangle_points[1], cv::Scalar(0, 233, 233), 3);
						}
					} 
					else
					{
						if (distanceX>-distanceY)
						{
							++left_count;
							down_count = 0;
							right_count = 0;
							up_count = 0;
							triangle_points[0].x = center.x + radius*1.25;
							triangle_points[0].y = center.y;
							triangle_points[1].x = triangle_points[2].x = center.x + radius;
							triangle_points[1].y = center.y + radius;
							triangle_points[2].y = center.y - radius;
							cv::line(img, triangle_points[0], triangle_points[1], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[0], triangle_points[2], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[2], triangle_points[1], cv::Scalar(0, 233, 233), 3);
						} 
						else
						{
							++up_count;
							down_count = 0;
							left_count = 0;
							right_count = 0;
							triangle_points[0].y = center.y - radius*1.25;
							triangle_points[0].x = center.x;
							triangle_points[1].y = triangle_points[2].y = center.y - radius;
							triangle_points[1].x = center.x + radius;
							triangle_points[2].x = center.x - radius;
							cv::line(img, triangle_points[0], triangle_points[1], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[0], triangle_points[2], cv::Scalar(0, 233, 233), 3);
							cv::line(img, triangle_points[2], triangle_points[1], cv::Scalar(0, 233, 233), 3);
						}
					}

					if (up_count == 8)
					{
						keybd_event(VK_UP , 0xE048, KEYEVENTF_EXTENDEDKEY | 0, 0);
						keybd_event(VK_UP , 0xE048, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
						//Sleep(1000);
						cv::fillConvexPoly(img, triangle_points, 3, cv::Scalar(0, 211, 34));
						up_count = 0;
					} 
					else if (down_count == 8)
					{
						keybd_event(VK_DOWN , 0xE051, KEYEVENTF_EXTENDEDKEY | 0, 0);
						keybd_event(VK_DOWN , 0xE051, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
						//Sleep(1000);
						cv::fillConvexPoly(img, triangle_points, 3, cv::Scalar(0, 211, 34));
						down_count = 0;
					} 
					else if (right_count == 8)
					{
						keybd_event(VK_RIGHT , 0xE04D, KEYEVENTF_EXTENDEDKEY | 0, 0);
						keybd_event(VK_RIGHT , 0xE04D, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
						//Sleep(1000);
						cv::fillConvexPoly(img, triangle_points, 3, cv::Scalar(0, 211, 34));
						right_count = 0;
					} 
					else if (left_count == 8)
					{
						keybd_event(VK_LEFT , 0xE04B, KEYEVENTF_EXTENDEDKEY | 0, 0);
						keybd_event(VK_LEFT , 0xE04B, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
						//Sleep(1000);
						cv::fillConvexPoly(img, triangle_points, 3, cv::Scalar(0, 211, 34));
						left_count = 0;
					}

					if (up_count >= 6)
					{
						cv::fillConvexPoly(img, triangle_points, 3, cv::Scalar(0, 211, 34));
					} 
					else if (down_count >= 6)
					{
						cv::fillConvexPoly(img, triangle_points, 3, cv::Scalar(0, 211, 34));
					} 
					else if (right_count >= 6)
					{
						cv::fillConvexPoly(img, triangle_points, 3, cv::Scalar(0, 211, 34));
					} 
					else if (left_count >= 6)
					{
						cv::fillConvexPoly(img, triangle_points, 3, cv::Scalar(0, 211, 34));
					}

				}
			}
		} 
		else
		{
			++lost_count;
			//first_detected_five_count = -10;
			if (control.hand == 2)
			{
				++fist_count;
				if (fist_count>0)
				{
					test.drawRing(img, cv::Scalar(233,34,111));
				}
				
			}
			else
			{
				test.draw_index = 1;
				fist_count = 0;
			}
			if (test.draw_index == num_segment)
			{
				keybd_event(VK_RETURN , 0x1C, KEYEVENTF_EXTENDEDKEY | 0, 0);
				keybd_event(VK_RETURN , 0x1C, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
				//Sleep(3000);
				//keybd_event(VK_DELETE , 0xE053, KEYEVENTF_EXTENDEDKEY | 0, 0);
				//keybd_event(VK_DELETE , 0xE053, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);

				fist_count = -20;
			}
			if (lost_count>10)
			{
				first_detected_five_count = -2;
			}
			else
			{
				cv::circle(img, center, radius, cv::Scalar(0,233,233), 3);
			}
		}
		endTime = clock();
		std::cout<<(double)(endTime - beginTime)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
		//cv::circle(img, cv::Point(320, 240), 50, cv::Scalar(9,9,255));
		cv::flip(img, img, 1);
		cv::imshow("result", img);
		cv::waitKey(20);
	}
	return 0;
}