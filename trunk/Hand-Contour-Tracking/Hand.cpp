#include "Hand.h"

const float Hand::interval = 0.04;
const int Hand::nknots[nbsplines] = {7,16,16,16,16,7,7,10,7,7};
const int Hand::nspans[nbsplines] = {2,7,7,7,7,2,2,5,2,2};
const int Hand::knotCounts[nbsplines][8] = 
{
	{3,1,3},{3,2,2,1,1,2,2,3},{3,2,2,1,1,2,2,3},{3,2,2,1,1,2,2,3},
	{3,2,2,1,1,2,2,3},{3,1,3},{3,1,3},{3,1,1,1,1,3},{3,1,3},{3,1,3}
};

const float score[22] = {0.5, 0.6408, 0.8738, 1.1767, 1.4805, 1.7232, 1.8801, 
	1.9610, 1.9922, 1.9995,2.0,2.0, 1.9995, 1.9922, 1.9610, 1.8801, 1.7232,
	1.4805, 1.1767, 0.8738, 0.6408, 0.5};

Hand::Hand(const string confile, const string ctrpntfile, const string mlinefile)
	//:a(0),s(1),sp(1),ap(0),w(1)
{
	loadTXT(confile,ctrpntfile,mlinefile);
}

Hand::Hand(const Hand& hand)
	:/*a(hand.a),s(hand.s),sp(hand.sp),ap(hand.ap),w(hand.w),*/
	handArea(hand.handArea),Gravity(hand.Gravity)
{
	PivotPoints = hand.PivotPoints.clone();
	TipPoints = hand.TipPoints.clone();

	MeasurePoints.reserve(nbsplines);
	ControlPoints.reserve(nbsplines);
	MeasureLinePoints.reserve(nbsplines);

	for (int i=0; i<nbsplines; i++)
	{
		MeasurePoints.push_back(hand.MeasurePoints[i].clone());
		ControlPoints.push_back(hand.ControlPoints[i].clone());
		int n = hand.MeasureLinePoints[i].size();
		vector<cv::Mat> PerMeasureLine;
		PerMeasureLine.reserve(n);
		for (int j=0; j<n; j++)
		{
			PerMeasureLine.push_back(hand.MeasureLinePoints[i][j].clone());
		}
		MeasureLinePoints.push_back(PerMeasureLine);
	}
}

Hand& Hand::operator=(const Hand &rhs)
{
	//a = rhs.a;
	//s = rhs.s;

	//ap = rhs.ap;
	//sp = rhs.sp;

	//w = rhs.w;

	handArea = rhs.handArea;
	Gravity = rhs.Gravity;
	PivotPoints = rhs.PivotPoints.clone();
	TipPoints = rhs.TipPoints.clone();

	MeasurePoints.clear();
	ControlPoints.clear();
	MeasureLinePoints.clear();
	MeasurePoints.reserve(nbsplines);
	ControlPoints.reserve(nbsplines);
	MeasureLinePoints.reserve(nbsplines);

	for (int i=0; i<nbsplines; i++)
	{
		MeasurePoints.push_back(rhs.MeasurePoints[i].clone());
		ControlPoints.push_back(rhs.ControlPoints[i].clone());
		int n = rhs.MeasureLinePoints[i].size();
		vector<cv::Mat> PerMeasureLine;
		PerMeasureLine.reserve(n);
		for (int j=0; j<n; j++)
		{
			PerMeasureLine.push_back(rhs.MeasureLinePoints[i][j].clone());
		}
		MeasureLinePoints.push_back(PerMeasureLine);
	}

	return *this;
}

Hand::~Hand()
{
}
bool Hand::loadTXT(const string confile, const string ctrpntfile, const string mlinefile)
{
	std::ifstream config(confile.c_str(), ifstream::in);

	if (!config)
	{
		std::cerr << "ERROR: can not open config file.";
		return false;
	}

	string s;

	config>>s;
	if (s != "-gravity")
	{
		std::cerr<<"ERROR: the config file is not in the right format.";
		return false;
	}

	config>>Gravity.x;
	config>>Gravity.y;
	//PreGravity.x = Gravity.x;
	//PreGravity.y = Gravity.y;

	config>>s;
	if ( s != "-pivotpoint")
	{
		std::cerr<<"ERROR: the config file is not in the right format.";
		return false;
	}
	PivotPoints.create(6, 1, CV_32FC2);
	for (int i = 0; i < 6; i++)
	{
		config>> *PivotPoints.ptr<float>(i);
		config>> *(PivotPoints.ptr<float>(i)+1);
	}

	config>>s;
	if ( s != "-tippoint")
	{
		std::cerr<<"ERROR: the config file is not in the right format.";
		return false;
	}
	TipPoints.create(6, 1, CV_32FC2);
	for (int i = 0; i < 6; i++)
	{
		config>> *TipPoints.ptr<float>(i);
		config>> *(TipPoints.ptr<float>(i)+1);
	}

	config>>s;
	if ( s != "-corner")
	{
		std::cerr<<"ERROR: the config file is not in the right format.";
		return false;
	}

	int x1, x2;
	int y1, y2;
	config>>x1>>y1>>x2>>y2;
	handArea = (x2-x1)*(y2-y1);

	config.close();

	std::ifstream ctrpnt(ctrpntfile.c_str());
	if (!ctrpnt)
	{
		std::cerr << "ERROR: can not open control point file.";
		return false;
	}
	std::ifstream mline(mlinefile.c_str());
	if (!mline)
	{
		std::cerr << "ERROR: can not open measurement line file.";
		return false;
	}

	//input the control points the knot count and the measure points
	ControlPoints.clear(); 
	MeasurePoints.clear();
	for (int i =0; i < nbsplines; i++)
	{
		int _nspans = nspans[i];
		cv::Mat ctrPoint(_nspans+1,1,CV_32FC2);

		for (int j = 0; j< _nspans+1; j++)
		{
			ctrpnt>>*ctrPoint.ptr<float>(j);
			ctrpnt>>*(ctrPoint.ptr<float>(j)+1);

		}
		ControlPoints.push_back(ctrPoint);

		mline>>s;
		//the num of measure points on each bspline
		int num_points;
		
		if ( s == "-n")
		{
			mline>>num_points;
			cv::Mat mPoints(num_points, 1, CV_32FC2);
			for (int k=0; k<num_points; k++)
			{
				mline>>*mPoints.ptr<float>(k);
				mline>>*(mPoints.ptr<float>(k)+1);
			}
			MeasurePoints.push_back(mPoints);
		}
		else
		{
			std::cerr << "ERROR: the measure line file is not in the right format.";
			return false;
		}
	}

	calAllMeasureLinePoints();

	return true;
}

void Hand::showPoint(cv::Mat& img, const vector<cv::Mat> sp, const Scalar& color, const int thickness)
{
	for (size_t i = 0; i < sp.size(); i++)
	{
		for (int j = 0; j < sp[i].rows; j++)
		{
			Point center;

			center.x = (int)*sp[i].ptr<float>(j);
			center.y = (int)*(sp[i].ptr<float>(j)+1);

			circle(img, center, 0, color, thickness);
		}
	}
}

void Hand::showHand(cv::Mat& img, const cv::Scalar color, const int thinckness)
{
	vector< vector<cv::Mat> > SpanPoints;
	SpanPoints.clear();
	SpanPoints.reserve(nbsplines);
	for (int i = 0; i<nbsplines; i++)
	{
		Bspline bspline(nspans[i], nknots[i], order, knotCounts[i], ControlPoints[i], interval);
		bspline.calSpanMatrices();
		bspline.calShowPoints();
		SpanPoints.push_back(bspline.getSpanPoints());
		showPoint(img, SpanPoints[i], color, thinckness);
	}
}

void Hand::showControlPoints(cv::Mat& img, const cv::Scalar color, const int thinckness)
{
	showPoint(img, ControlPoints, color, thinckness);
}

void Hand::showMeasurePoints(cv::Mat& img, const cv::Scalar color, const int thinckness)
{
	showPoint(img, MeasurePoints, color, thinckness);
}

void Hand::calMeasureSlope(vector< vector <float> > &Measureslope)
{
	CV_Assert( nbsplines == MeasurePoints.size());
	Measureslope.reserve(nbsplines);
	for (int i=0; i<nbsplines; i++)
	{
		int num_MeasurePoints = MeasurePoints[i].rows;
		vector<float> _MeasureSlope;
		_MeasureSlope.reserve(num_MeasurePoints);
		float dx, dy;
		switch (i)
		{
		case 0:
		case 5:
		case 7:
		case 9:
			for (int j = 1; j<num_MeasurePoints-1; j++)
			{
				float dx, dy;
				dx = *MeasurePoints[i].ptr<float>(j-1) - *(MeasurePoints[i].ptr<float>(j+1));
				dy = *(MeasurePoints[i].ptr<float>(j-1)+1) - *(MeasurePoints[i].ptr<float>(j+1)+1);
				if ( j==1 || j==num_MeasurePoints-2)
				{
					_MeasureSlope.push_back(-1*dx/dy);
					_MeasureSlope.push_back(-1*dx/dy);
				}
				else
				{
					_MeasureSlope.push_back(-1*dx/dy);
				}
			}
			Measureslope.push_back(_MeasureSlope);
			break;
		case 1:
		case 2:
		case 3:
		case 4:
			dx = *PivotPoints.ptr<float>(i-1) - *TipPoints.ptr<float>(i-1);
			dy = *(PivotPoints.ptr<float>(i-1)+1) - *(TipPoints.ptr<float>(i-1)+1);
			for (int j = 0; j<num_MeasurePoints; j++)
			{
				if ( j == 4 || j == 5)
				{
					_MeasureSlope.push_back(dy/dx);
				} 
				else
				{
					_MeasureSlope.push_back(-1*dx/dy);
				}
			}
			Measureslope.push_back(_MeasureSlope);
			break;
		case 6:
			dx = *PivotPoints.ptr<float>(i-2) - *TipPoints.ptr<float>(i-2);
			dy = *(PivotPoints.ptr<float>(i-2)+1) - *(TipPoints.ptr<float>(i-2)+1);
			for (int j = 0; j<num_MeasurePoints; j++)
			{
				_MeasureSlope.push_back(-1*dx/dy);
			}
			Measureslope.push_back(_MeasureSlope);
			break;
		case 8:
			dx = *PivotPoints.ptr<float>(i-4) - *TipPoints.ptr<float>(i-4);
			dy = *(PivotPoints.ptr<float>(i-4)+1) - *(TipPoints.ptr<float>(i-4)+1);
			for (int j = 0; j<num_MeasurePoints; j++)
			{
				_MeasureSlope.push_back(-1*dx/dy);
			}
			Measureslope.push_back(_MeasureSlope);
			break;
		}
	}
}

void Hand::calLinePoints(cv::Mat &line, int x0, int y0, float k, int begin, int end, bool d, bool d2)
{
	float error = 0;
	int num = 0;
	bool a = true;
	bool neg = false;
	if(k<0) neg = true;
	k = abs(k);
	if(k>1)
	{
		std::swap(x0,y0);
		k = 1/k;
		a = false;
	}

	int y = y0;

	int n = end -begin + 1;
	for(int x = x0 ; num<n; num++)
	{
		if(a)
		{
			if (d2)
			{
				*line.ptr<float>(begin + num) = x;
				*(line.ptr<float>(begin +num)+1) = y;
			} 
			else
			{
				*line.ptr<float>(end - num) = x;
				*(line.ptr<float>(end - num)+1) = y;
			}
		}
		else
		{
			if (d2)
			{
				*line.ptr<float>(begin + num) = y;
				*(line.ptr<float>(begin +num)+1) = x;
			} 
			else
			{
				*line.ptr<float>(end - num) = y;
				*(line.ptr<float>(end - num)+1) = x;
			}
		}
		error += k;
		if(abs(error)>=0.5)
		{
			if (neg)
			{
				if(!d)y++;
				else y--;
			} 
			else
			{
				if(!d)y--;
				else y++;
			}
			error--;
		}
		if(!d) x--;
		else x++;

	}
}
//{
//	float error = 0;
//	int num = 0;
//	bool a = true;
//	bool neg = false;
//	if(k<0) neg = true;
//	k = abs(k);
//	if(k>1)
//	{
//		std::swap(x0,y0);
//		k = 1/k;
//		a = false;
//	}
//
//	int y = y0;
//
//
//
//	int n = end -begin + 1;
//	for(int x = x0 ; num<n; num++)
//	{
//		if(a)
//		{
//			
//			*line.ptr<float>(begin + num) = x;
//			*(line.ptr<float>(begin +num)+1) = y;
//		}
//		else
//		{
//			*line.ptr<float>(begin + num) = y;
//			*(line.ptr<float>(begin +num)+1) = x;
//		}
//		error += k;
//		if(abs(error)>=0.5)
//		{
//			if (neg)
//			{
//				if(!d)y++;
//				else y--;
//			} 
//			else
//			{
//				if(!d)y--;
//				else y++;
//			}
//			error--;
//		}
//		if(!d) x--;
//		else x++;
//
//	}
//}


void Hand::calAllMeasureLinePoints()
//{
//	vector< vector<float> > Measureslope;
//	calMeasureSlope(Measureslope);
//	MeasureLinePoints.reserve(nbsplines);
//	for (int i = 0; i<nbsplines; i++)
//	{
//		vector<cv::Mat> MeasurePointsPerSpline;
//		int num_measure_points_per_spline = MeasurePoints[i].rows;
//		MeasurePointsPerSpline.reserve(num_measure_points_per_spline);
//		int j = 0;
//		switch(i)
//		{
//		case 0:
//		case 6:
//			for (j = 0; j<num_measure_points_per_spline; j++)
//			{
//				Mat line_points(22,1,CV_32FC2);
//				int x = *MeasurePoints[i].ptr<float>(j);
//				int y = *(MeasurePoints[i].ptr<float>(j)+1);
//				float k = Measureslope[i][j];
//				calLinePoints(line_points, x, y, k, 0, 10, false, false);
//				calLinePoints(line_points, x, y, k, 11, 21, true, true);
//				MeasurePointsPerSpline.push_back(line_points);	
//			}
//			break;
//		case 1:
//		case 2:
//		case 3:
//		case 4:		
//			for (j = 0; j<6; j++)
//			{
//				Mat line_points(22,1,CV_32FC2);
//				int x = *MeasurePoints[i].ptr<float>(j);
//				int y = *(MeasurePoints[i].ptr<float>(j)+1);
//				float k = Measureslope[i][j];
//				calLinePoints(line_points, x, y, k, 0, 10, false,false);
//				calLinePoints(line_points, x, y, k, 11, 21, true,true);
//				MeasurePointsPerSpline.push_back(line_points);	
//			}
//			for (; j<num_measure_points_per_spline; j++)
//			{
//				Mat line_points(22,1,CV_32FC2);
//				int x = *MeasurePoints[i].ptr<float>(j);
//				int y = *(MeasurePoints[i].ptr<float>(j)+1);
//				float k = Measureslope[i][j];
//				calLinePoints(line_points, x, y, k, 0, 10, true,false);
//				calLinePoints(line_points, x, y, k, 11, 21, false,true);
//				MeasurePointsPerSpline.push_back(line_points);
//			}
//			break;
//		case 5:
//		case 8:
//		case 9:
//			for (j = 0; j<num_measure_points_per_spline; j++)
//			{
//				Mat line_points(22,1,CV_32FC2);
//				int x = *MeasurePoints[i].ptr<float>(j);
//				int y = *(MeasurePoints[i].ptr<float>(j)+1);
//				float k = Measureslope[i][j];
//				calLinePoints(line_points, x, y, k, 0, 10, true,false);
//				calLinePoints(line_points, x, y, k, 11, 21, false,true);
//				MeasurePointsPerSpline.push_back(line_points);	
//			}
//			break;
//		case 7:
//			for (j = 0; j<3; j++)
//			{
//				Mat line_points(22,1,CV_32FC2);
//				int x = *MeasurePoints[i].ptr<float>(j);
//				int y = *(MeasurePoints[i].ptr<float>(j)+1);
//				float k = Measureslope[i][j];
//				calLinePoints(line_points, x, y, k, 0, 10, false,false);
//				calLinePoints(line_points, x, y, k, 11, 21, true,true);
//				MeasurePointsPerSpline.push_back(line_points);	
//			}
//			for (; j<num_measure_points_per_spline; j++)
//			{
//				Mat line_points(22,1,CV_32FC2);
//				int x = *MeasurePoints[i].ptr<float>(j);
//				int y = *(MeasurePoints[i].ptr<float>(j)+1);
//				float k = Measureslope[i][j];
//				calLinePoints(line_points, x, y, k, 0, 10, true,false);
//				calLinePoints(line_points, x, y, k, 11, 21, false,true);
//				MeasurePointsPerSpline.push_back(line_points);
//			}
//			break;
//		}
//		//for (int j = 0; j<num_measure_points_per_spline; j++)
//		//{
//		//	Mat line_points(11,1,CV_32FC2);
//		//	int x = *MeasurePoints[i].ptr<float>(j);
//		//	int y = *(MeasurePoints[i].ptr<float>(j)+1);
//		//	float k = Measureslope[i][j];
//		//	calLinePoints(line_points, x, y, k, 0, 10, true);
//		//	//calLinePoints(line_points, x, y, k, 11, 21, false);
//		//	MeasurePointsPerSpline.push_back(line_points);	
//		//}
//		MeasureLinePoints.push_back(MeasurePointsPerSpline);
//	}
//}

{
	vector< vector<float> > Measureslope;
	calMeasureSlope(Measureslope);
	MeasureLinePoints.reserve(nbsplines);
	for (int i = 0; i<nbsplines; i++)
	{
		vector<cv::Mat> MeasurePointsPerSpline;
		int num_measure_points_per_spline = MeasurePoints[i].rows;
		MeasurePointsPerSpline.reserve(num_measure_points_per_spline);
		int j = 0;
		switch(i)
		{
		case 0:
		case 6:
			for (j = 0; j<num_measure_points_per_spline; j++)
			{
				Mat line_points(22,1,CV_32FC2);
				int x = *MeasurePoints[i].ptr<float>(j);
				int y = *(MeasurePoints[i].ptr<float>(j)+1);
				float k = Measureslope[i][j];
				calLinePoints(line_points, x, y, k, 0, 10, false, false);
				//calLinePoints(line_points, x, y, k, 0, 10, true, true);
				calLinePoints(line_points, x, y, k, 11, 21, true, true);
				MeasurePointsPerSpline.push_back(line_points);	
			}
			break;
		case 1:
		case 2:
		case 3:
		case 4:		
			for (j = 0; j<6; j++)
			{
				Mat line_points(22,1,CV_32FC2);
				int x = *MeasurePoints[i].ptr<float>(j);
				int y = *(MeasurePoints[i].ptr<float>(j)+1);
				float k = Measureslope[i][j];
				calLinePoints(line_points, x, y, k, 0, 10, false,false);
				//calLinePoints(line_points, x, y, k, 0, 10, true, true);
				calLinePoints(line_points, x, y, k, 11, 21, true,true);
				MeasurePointsPerSpline.push_back(line_points);	
			}
			for (; j<num_measure_points_per_spline; j++)
			{
				Mat line_points(22,1,CV_32FC2);
				int x = *MeasurePoints[i].ptr<float>(j);
				int y = *(MeasurePoints[i].ptr<float>(j)+1);
				float k = Measureslope[i][j];
				calLinePoints(line_points, x, y, k, 0, 10, true,false);
				//calLinePoints(line_points, x, y, k, 0, 10, false, true);
				calLinePoints(line_points, x, y, k, 11, 21, false,true);
				MeasurePointsPerSpline.push_back(line_points);
			}
			break;
		case 5:
		case 8:
		case 9:
			for (j = 0; j<num_measure_points_per_spline; j++)
			{
				Mat line_points(22,1,CV_32FC2);
				int x = *MeasurePoints[i].ptr<float>(j);
				int y = *(MeasurePoints[i].ptr<float>(j)+1);
				float k = Measureslope[i][j];
				calLinePoints(line_points, x, y, k, 0, 10, true,false);
				//calLinePoints(line_points, x, y, k, 0, 10, false, true);
				calLinePoints(line_points, x, y, k, 11, 21, false,true);
				MeasurePointsPerSpline.push_back(line_points);	
			}
			break;
		case 7:
			for (j = 0; j<3; j++)
			{
				Mat line_points(22,1,CV_32FC2);
				int x = *MeasurePoints[i].ptr<float>(j);
				int y = *(MeasurePoints[i].ptr<float>(j)+1);
				float k = Measureslope[i][j];
				calLinePoints(line_points, x, y, k, 0, 10, false,false);
				//calLinePoints(line_points, x, y, k, 0, 10, true, true);
				calLinePoints(line_points, x, y, k, 11, 21, true,true);
				MeasurePointsPerSpline.push_back(line_points);	
			}
			for (; j<num_measure_points_per_spline; j++)
			{
				Mat line_points(22,1,CV_32FC2);
				int x = *MeasurePoints[i].ptr<float>(j);
				int y = *(MeasurePoints[i].ptr<float>(j)+1);
				float k = Measureslope[i][j];
				calLinePoints(line_points, x, y, k, 0, 10, true,false);
				//calLinePoints(line_points, x, y, k, 0, 10, false, true);
				calLinePoints(line_points, x, y, k, 11, 21, false,true);
				MeasurePointsPerSpline.push_back(line_points);
			}
			break;
		}
		//for (int j = 0; j<num_measure_points_per_spline; j++)
		//{
		//	Mat line_points(11,1,CV_32FC2);
		//	int x = *MeasurePoints[i].ptr<float>(j);
		//	int y = *(MeasurePoints[i].ptr<float>(j)+1);
		//	float k = Measureslope[i][j];
		//	calLinePoints(line_points, x, y, k, 0, 10, true);
		//	//calLinePoints(line_points, x, y, k, 11, 21, false);
		//	MeasurePointsPerSpline.push_back(line_points);	
		//}
		MeasureLinePoints.push_back(MeasurePointsPerSpline);
	}
}




void Hand::showMeasureLinePoints(cv::Mat& img, const cv::Scalar color, const int thinckness)
{
	for (int i = 0; i<nbsplines; i++)
	{
		showPoint(img, MeasureLinePoints[i], color, thinckness);
	}
}

void Hand::affineHand(int x, int y, float scale, float angle)
{
	Gravity.x = Gravity.x + x;
	Gravity.y = Gravity.y + y;

	handArea = handArea * scale;

	PivotPoints = PivotPoints + cv::Scalar(x,y);
	TipPoints = TipPoints + cv::Scalar(x,y);

	cv::Mat affineMat = cv::getRotationMatrix2D(Gravity,angle,scale);

	cv::transform(PivotPoints, PivotPoints, affineMat);
	cv::transform(TipPoints, TipPoints, affineMat);

	for (int i = 0; i<nbsplines; i++)
	{
		cv::Mat preMeasurePoint = MeasurePoints[i].clone();
		ControlPoints[i] = ControlPoints[i] + cv::Scalar(x,y);
		MeasurePoints[i] = MeasurePoints[i] + cv::Scalar(x,y);
		cv::transform(ControlPoints[i], ControlPoints[i], affineMat);
		cv::transform(MeasurePoints[i], MeasurePoints[i], affineMat);
		int n = MeasureLinePoints[i].size();
		for (int j = 0; j<n; j++)
		{
			float dx, dy, x0, y0;
			x0 = *MeasurePoints[i].ptr<float>(j);
			y0 = *(MeasurePoints[i].ptr<float>(j)+1);
			dx = x0 - *preMeasurePoint.ptr<float>(j);
			dy = y0 - *(preMeasurePoint.ptr<float>(j)+1);
			cv::Mat affineLineMat = cv::getRotationMatrix2D(cv::Point2f(x0,y0),angle,scale);
			MeasureLinePoints[i][j] = MeasureLinePoints[i][j] + cv::Scalar(dx,dy);
			cv::transform(MeasureLinePoints[i][j], MeasureLinePoints[i][j],affineLineMat);
		}
	}
}

void Hand::affineFinger(FINGER finger, float scale, float angle)
{
	float x0, y0;
	x0 = *PivotPoints.ptr<float>(finger-1);
	y0 = *(PivotPoints.ptr<float>(finger-1)+1);
	cv::Mat affineMat = cv::getRotationMatrix2D(cv::Point2f(x0,y0),angle,1);
	cv::transform(ControlPoints[finger], ControlPoints[finger],affineMat);
	cv::transform(MeasurePoints[finger], MeasurePoints[finger],affineMat);
	int n = MeasureLinePoints[finger].size();
	for (int i = 0; i<n; i++)
	{
		cv::transform(MeasureLinePoints[finger][i], MeasureLinePoints[finger][i], affineMat);
	}
	float x1, y1;
	x1 = *TipPoints.ptr<float>(finger-1);
	y1 = *(TipPoints.ptr<float>(finger-1)+1);
	float alpha = 1/sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1));
	cv::Point2f v(alpha*(x0-x1),alpha*(y0-y1));

	if (scale == 1)
		return;

	float distance;
	float x, y;
	int i = 0;

	for (; i<4; i++)
	{
		x = *MeasurePoints[finger].ptr<float>(i) - *ControlPoints[finger].ptr<float>(0);
		y = *(MeasurePoints[finger].ptr<float>(i)+1) - *(ControlPoints[finger].ptr<float>(0)+1);
		distance = (1-scale)*sqrt(x*x + y*y);
		*MeasurePoints[finger].ptr<float>(i) += distance*v.x;
		*(MeasurePoints[finger].ptr<float>(i)+1) += distance*v.y;
		MeasureLinePoints[finger][i] += cv::Scalar(distance*v.x, distance*v.y);

		x = *ControlPoints[finger].ptr<float>(i) - *ControlPoints[finger].ptr<float>(0);
		y = *(ControlPoints[finger].ptr<float>(i)+1) - *(ControlPoints[finger].ptr<float>(0)+1);
		distance = (1-scale)*sqrt(x*x + y*y);
		*ControlPoints[finger].ptr<float>(i) += distance*v.x;
		*(ControlPoints[finger].ptr<float>(i)+1) += distance*v.y;	
	}

	for (; i<6; i++)
	{
		*MeasurePoints[finger].ptr<float>(i) += distance*v.x;
		*(MeasurePoints[finger].ptr<float>(i)+1) += distance*v.y;
		MeasureLinePoints[finger][i] += cv::Scalar(distance*v.x, distance*v.y);
	}

	for (; i<10; i++)
	{
		x = *ControlPoints[finger].ptr<float>(i-2) - *ControlPoints[finger].ptr<float>(7);
		y = *(ControlPoints[finger].ptr<float>(i-2)+1) - *(ControlPoints[finger].ptr<float>(7)+1);
		distance = (1-scale)*sqrt(x*x + y*y);
		*ControlPoints[finger].ptr<float>(i-2) += distance*v.x;
		*(ControlPoints[finger].ptr<float>(i-2)+1) += distance*v.y;

		x = *MeasurePoints[finger].ptr<float>(i) - *ControlPoints[finger].ptr<float>(7);
		y = *(MeasurePoints[finger].ptr<float>(i)+1) - *(ControlPoints[finger].ptr<float>(7)+1);
		distance = (1-scale)*sqrt(x*x + y*y);
		*MeasurePoints[finger].ptr<float>(i) += distance*v.x;
		*(MeasurePoints[finger].ptr<float>(i)+1) += distance*v.y;
		MeasureLinePoints[finger][i] += cv::Scalar(distance*v.x, distance*v.y);
	}
}

float Hand::calWeight(cv::Mat& img)
{
	CV_Assert(img.channels() == 1);
	float w = 1;
	//int count =0;

	for (int i=0; i<nbsplines; i++)
	{
		int n = MeasureLinePoints[i].size();
		float _w = 1;
		for (int j=0; j<n; j++)
		{
			int k = 0;
			for ( ; k<21; k++)
			{
				int x0, y0, x1, y1;
				x0 = *MeasureLinePoints[i][j].ptr<float>(k);
				y0 = *(MeasureLinePoints[i][j].ptr<float>(k)+1);
				x1 = *MeasureLinePoints[i][j].ptr<float>(k+1);
				y1 = *(MeasureLinePoints[i][j].ptr<float>(k+1)+1);
				if (img.at<uchar>(y0,x0)==255 && img.at<uchar>(y1,x1)==255)
					break;
			}
			_w *= score[k];
		}
		w *=_w;
	}
	return w;
}