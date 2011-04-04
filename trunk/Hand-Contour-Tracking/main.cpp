#include "Bspline.h"
#include "Hand.h"
#include "opencv.h"
#include "TSLskinSeg.h"
#include "particles.h"
#include <ctime>


int main()
{
	string confile = "config4.txt";
	string cntpntfile = "controlPoint2.txt";
	string mlinefile = "measuremenline.txt";

	cv::Mat img1 = cv::imread("ii.jpg");
	
	if (img1.data == NULL) 
	{
		std::cout<<"can't read the image!"<<std::endl;
	}

	Hand hand(confile, cntpntfile, mlinefile);
	hand.affineHand(0,-250,0.45,0);
	//hand.affineFinger(Hand::INDEX,0.35,0);
	//Hand hand2(hand);

	cv::RNG rng(time(NULL));
	particle *particles, *new_particles;

	hand.showHand(img1,Scalar(0,0,0),2);
	hand.showMeasureLinePoints(img1, cv::Scalar(255,0,255),2);
	//hand.showControlPoints(img1,Scalar(147,21,128), 8);
	//imshow("tst",img1);

	VideoCapture cap(1);
	
	
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	if (!cap.isOpened())
	{
		cout<<"can't open the video camera!"<<endl;
		return -1;
	}
	
	int num_particles = 500;

	particles = init_distribution(hand.GetGravity(), num_particles);

	Mat img;
	//namedWindow("original",CV_WINDOW_AUTOSIZE);
	cap>>img;
	int width, height;
	width = img.cols;
	height = img.rows;
	Mat element = Mat::ones(5,5,CV_8UC1);
	Mat BW = Mat::zeros(width*2,height*2,CV_8UC1);
	Mat img_bw(BW,cv::Rect(width/2,height/2,width,height));
	//Mat img_bw = Mat::zeros(img.size(),CV_8UC1);
	particle preParitcle;
	preParitcle.w = 0;
	preParitcle.x = hand.GetGravity().x;
	preParitcle.y = hand.GetGravity().y;
	preParitcle.s = 1;
	preParitcle.a = 0;

	while (1)
	{
		cap>>img;
		TSLskinSegment(img,img_bw);
		medianBlur(img_bw,img_bw,7);

		hand.showHand(img,Scalar(255,255,0),3);
		//hand.showMeasureLinePoints(img, cv::Scalar(255,0,255),2);
		cv::flip(img,img,1);
		imshow("original",img);

		if (waitKey(20)>=0)
		{
			break;
		}
		//cout<<hand.calWeight(img_bw)<<endl;
		if(hand.calWeight(img_bw)>1) break;
	}


	bool keep = false;


	int sum = 0;
	while (1)
	{


		stringstream   stream;
		string name;
		stream<<sum;
		sum++;
		name = stream.str();
		name = name+".jpg";


		cap>>img;
		//medianBlur(img, img, 7);


		double t = (double)cvGetTickCount();
		TSLskinSegment(img,img_bw);
		//TSLskinSegment(img_roi, img_bw_roi);

		medianBlur(img_bw,img_bw,7);
		//morphologyEx(img_bw, img_bw, MORPH_OPEN, element);

		//double t = (double)cvGetTickCount();

		int x, y;
		float s, a;

		if (!keep)
		{
#pragma omp parallel for
			for (int i = 0; i<num_particles; i++)
			{

				//std::cout<<i<<std::endl;
				Hand hand2(hand);

				particles[i] = transition(particles[i], width, height, rng);

				x = cvRound(particles[i].x - hand.GetGravity().x);
				y = cvRound(particles[i].y - hand.GetGravity().y);
				s = particles[i].s;
				a = particles[i].a;
				//std::cout<<"aa = "<<a<<std::endl;
				a = 180*a/PI;

				hand2.affineHand(x, y, s, a);
				//hand2.showHand(img,Scalar(255,0,255),1);

				particles[i].w = hand2.calWeight(img_bw);


				//if (hand2.w>1000)
				//{
				//	cout<<"weight:"<<hand2.w<<endl;
				//}

				//cout<<"weight:"<<hand2.w<<endl;
				//hand2.showHand(img, cv::Scalar(255,255,0),2);
			}
		}
		//#pragma omp parallel for
		//for (int i = 0; i<num_particles; i++)
		//{

		//	//std::cout<<i<<std::endl;
		//	Hand hand2(hand);

		//	particles[i] = transition(particles[i], 550, 400, rng);

		//	x = cvRound(particles[i].x - hand.GetGravity().x);
		//	y = cvRound(particles[i].y - hand.GetGravity().y);
		//	s = particles[i].s;
		//	a = particles[i].a;
		//	//std::cout<<"aa = "<<a<<std::endl;
		//	a = 180*a/PI;

		//	hand2.affineHand(x, y, s, a);
		//	hand2.showHand(img,Scalar(255,0,255),1);

		//	particles[i].w = hand2.calWeight(img_bw);


		//	//if (hand2.w>1000)
		//	//{
		//	//	cout<<"weight:"<<hand2.w<<endl;
		//	//}

		//	//cout<<"weight:"<<hand2.w<<endl;
		//	//hand2.showHand(img, cv::Scalar(255,255,0),2);
		//}

		Hand hand2(hand);
		x = cvRound(preParitcle.x - hand.GetGravity().x);
		y = cvRound(preParitcle.y - hand.GetGravity().y);
		s = preParitcle.s;
		a = preParitcle.a;
		//std::cout<<"aa = "<<a<<std::endl;
		a = 180*a/PI;

		hand2.affineHand(x, y, s, a);

		preParitcle.w = hand2.calWeight(img_bw);

		qsort( particles, num_particles, sizeof( particle ), &particle_cmp );
		//std::cout<<"pre:"<<preParitcle.w<<std::endl;
		//std::cout<<"now:"<<particles[0].w<<std::endl;
		particle tmp = particles[0];
		if (preParitcle.w < particles[0].w)
		{
			normalize_weights(particles, num_particles);

			new_particles = resample(particles, num_particles);

			free(particles);
			particles = new_particles;

			preParitcle = tmp;
			//std::cout<<"======="<<endl;
			keep = false;
		}
		else keep = true;
		/*normalize_weights(particles, num_particles);

		new_particles = resample(particles, num_particles);

		free(particles);
		particles = new_particles;*/


		x = cvRound(preParitcle.x - hand.GetGravity().x);
		y = cvRound(preParitcle.y - hand.GetGravity().y);
		s = preParitcle.s;
		a = preParitcle.a;
		//std::cout<<"aa = "<<a<<std::endl;
		a = 180*a/PI;



		//x = cvRound(particles[0].x - hand.GetGravity().x);
		//y = cvRound(particles[0].y - hand.GetGravity().y);
		//s = particles[0].s;
		//a = particles[0].a;

		//a = 180*a/PI;

		Hand hand3(hand);
		hand3.affineHand(x, y, s, a);
		hand3.showHand(img,Scalar(255,0,0),3);

		t = (double)cvGetTickCount()-t;
		printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
		//free(new_particles);
		//hand.calWeight(img_bw);
		//t = (double)cvGetTickCount()-t;
		
		

		//namedWindow("bw",CV_WINDOW_AUTOSIZE);

		//printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

		imshow("bw",img_bw);

		
		//hand.showMeasureLinePoints(img, cv::Scalar(255,0,255),2);
		cv::flip(img,img,1);
		imshow("original",img);

		//imwrite(name,img);

		if (waitKey(20)>=0)
		{
			break;
		}	
	}
	free(new_particles);
	//cv::Mat img = cv::imread("ii.jpg");

	//Hand hand(confile, cntpntfile, mlinefile);
	//Hand hand2(hand);
	////hand.calcHandBspline();
	//int nBpines = hand.GetNumBspine();

	////hand.showHand(img,cv::Scalar(255,122,0),3);
	////hand.showControlPoints(img, cv::Scalar(0,255,0),8);
	////hand.showMeasurePoints(img, cv::Scalar(0,0,255),8);
	////hand.showMeasureLinePoints(img, cv::Scalar(255,0,255),3);
	//double t = (double)cvGetTickCount();
	//for(int i = 0; i <1; i++)
	//{

	//	hand.showControlPoints(img, cv::Scalar(0,255,0),1);
	//	hand.showMeasurePoints(img, cv::Scalar(0,0,255),1);
	//	hand.showMeasureLinePoints(img, cv::Scalar(255,0,255),1);


	//	hand.affineHand(100,100,0.5,30);
	//	hand.affineFinger(Hand::INDEX,1,-10);
	//	hand.affineFinger(Hand::LITTLE,0.5,10);
	//	hand.affineFinger(Hand::MIDDLE,0.4,0);
	//	hand.affineFinger(Hand::RING,1.6,0);
	//}
	////hand.calcHandBspline();
	//t = (double)cvGetTickCount() - t;
	//cout << "Affine time = " << t*1000./cv::getTickFrequency() << "ms" << endl; 
	//hand.showHand(img, cv::Scalar(126,100,200),1);
	//hand.showControlPoints(img, cv::Scalar(0,255,0),1);
	//hand.showMeasurePoints(img, cv::Scalar(0,0,255),1);
	//hand.showMeasureLinePoints(img, cv::Scalar(255,0,255),1);

	//hand2.showHand(img, cv::Scalar(0,255,255),1);

	//hand2 = hand;
	//hand2.showHand(img, cv::Scalar(255,0,255),1);
	
	//for (int i=0; i<nBpines; i++)
	//{
	//	int num_measurePoint = hand.MeasurePoints[i].rows;
	//	for (int j=0; j< num_measurePoint; j++)
	//	{
	//		int x,y;
	//		float k;
	//		k = 1*hand.MeasureSlope[i][j];
	//		x = *hand.MeasurePoints[i].ptr<float>(j);
	//		y = *(hand.MeasurePoints[i].ptr<float>(j)+1);
	//		//Line(img,x,y,k,50,true);
	//	}
	//}

	//cv::imwrite("result.jpg", img1);
	//while (1)
	//{
	//	cv::imshow("img", img1);
	//	if (cv::waitKey(30) == 27)
	//	{
	//		break;
	//	}
	//}

	//cout<<hand.PivotPoints<<endl;
	//for (int i = 0; i < 6; i++)
	//{
	//	cout<< *hand.TipPoints.ptr<float>(i)<<endl;;
	//	cout<< *(hand.TipPoints.ptr<float>(i)+1)<<endl;;
	//}
	return 0;
}