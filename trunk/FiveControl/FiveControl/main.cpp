#include "FiveControl.h"
#include "drawCircle.h"
#include <ctime>
#include <cmath>
#include <Windows.h>

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

	while(1)
	{
		cap>>img;
		beginTime = clock();
		control.process(img);
		if (control.hand == 1)
		{
			//cv::rectangle(img, control.handLocation, cv::Scalar(0,0,255), 3);

			int x = control.handLocation.x + control.handLocation.width/2;
			int y = control.handLocation.y + control.handLocation.height/2;
			int distanceX = x - 320;
			int distanceY = y - 240;
			float distance = (float)sqrt((float)(distanceX*distanceX+
				distanceY*distanceY));
			if (distance<50)
			{
				test.drawRing(img, cv::Scalar(233,34,111));
				if (test.draw_index == 20)
				{
					/*keybd_event(VK_RETURN , 0x1C, KEYEVENTF_EXTENDEDKEY | 0, 0);
					keybd_event(VK_RETURN , 0x1C, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);*/
				}
			}
			else
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
					} 
					else
					{
						++down_count;
						up_count = 0;
						left_count = 0;
						right_count = 0;
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
					} 
					else
					{
						++down_count;
						up_count = 0;
						left_count = 0;
						right_count = 0;
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
					} 
					else
					{
						++up_count;
						down_count = 0;
						left_count = 0;
						right_count = 0;
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
					} 
					else
					{
						++up_count;
						down_count = 0;
						left_count = 0;
						right_count = 0;
					}
				}

				if (up_count == 8)
				{
					keybd_event(VK_UP , 0xE048, KEYEVENTF_EXTENDEDKEY | 0, 0);
					keybd_event(VK_UP , 0xE048, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					up_count = 0;
				} 
				else if (down_count == 8)
				{
					keybd_event(VK_DOWN , 0xE051, KEYEVENTF_EXTENDEDKEY | 0, 0);
					keybd_event(VK_DOWN , 0xE051, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					down_count = 0;
				} 
				else if (right_count == 8)
				{
					keybd_event(VK_RIGHT , 0xE04D, KEYEVENTF_EXTENDEDKEY | 0, 0);
					keybd_event(VK_RIGHT , 0xE04D, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					right_count = 0;
				} 
				else if (left_count == 8)
				{
					keybd_event(VK_LEFT , 0xE04B, KEYEVENTF_EXTENDEDKEY | 0, 0);
					keybd_event(VK_LEFT , 0xE04B, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
					left_count = 0;
				}

			}


		}
		else
		{
			//test.draw_index = 1;
			if (control.hand == 2)
			{
				++fist_count;
				test.drawRing(img, cv::Scalar(233,34,111));
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
				//keybd_event(VK_DELETE , 0xE053, KEYEVENTF_EXTENDEDKEY | 0, 0);
				//keybd_event(VK_DELETE , 0xE053, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
				fist_count = 0;
				//test.draw_index = 1;
			}
		}

		endTime = clock();
		std::cout<<(double)(endTime - beginTime)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
		cv::circle(img, cv::Point(320, 240), 50, cv::Scalar(9,9,255));
		cv::flip(img, img, 1);
		cv::imshow("result", img);
		cv::waitKey(20);
	}
	return 0;
}