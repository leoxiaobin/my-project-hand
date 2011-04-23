#include "FiveControl.h"

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

	//cv::Rect handLocation;
	//handLocation.x = 0;
	//handLocation.y = 0;
	//handLocation.width = img.cols;
	//handLocation.height = img.rows;
	//int hand = 0;

	while(1)
	{
		cap>>img;
		//control.process(img, hand, handLocation);
		control.process(img);
		if (control.hand == 1)
		{
			//cout<<"**************************************"<<endl;
			//cout<<"Find five in the image!"<<endl;
			cv::rectangle(img, control.handLocation, cv::Scalar(0,0,255), 3);
		}
		else if (control.hand == 2)
		{
			//cout<<"==============="<<endl;
			//cout<<"Find fist in the image!"<<endl;
			cv::rectangle(img, control.handLocation, cv::Scalar(0,255,0),3);
		}
		else
		{
			//cout<<"Find nothing in the image!"<<endl;
		}
		cv::imshow("result", img);
		cv::waitKey(20);
	}
	return 0;
}