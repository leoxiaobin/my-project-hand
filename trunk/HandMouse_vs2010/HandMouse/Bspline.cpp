#include "Bspline.h"
//Bspline::Bspline(void)
//{
//}
Bspline::~Bspline(void)
{
	delete [] knots;
	delete [] firstKnots;
}
//Bspline::Bspline(int _nspans, int _nknots, int _order, int*_knotsCount, Point* _contrlPoints, 
//	float _interval)
//{
//	nspans = _nspans;
//	nknots = _nknots;
//	order = _order;
//	knotsCount = _knotsCount;
//	contrlPoints = _contrlPoints;
//	interval = _interval;
//	knots = new int[_nknots];
//	firstKnots = new int[_nspans];
//	nfuncs = nknots - order;
//	ww = (int)(1.0 / _interval);
//}

Bspline::Bspline(const int _nspans, const int _nknots, const int _order, const int*  _knotsCount, Mat& _controlPoints,
	const  float _interval):nspans(_nspans), nknots(_nknots), order(_order), knotsCount(_knotsCount),
	interval(_interval)
{
	ContrlPoints = _controlPoints.clone();
	knots        = new int[_nknots];
	firstKnots   = new int[_nspans];
	nfuncs       = nknots - order;
	ww           = (int)(1.0 / _interval);
}


bool Bspline::calSpanMatrices()
{
	int i, j;

	///**************/
	//int p = 0;
	//int q = 0;
	//for (i = 0; i <= nspans; i++)
	//{
	//	for (j = 1; j <= knotsCount[i] ; j++)
	//	{
	//		knots[p] = q;
	//		p++;
	//	}
	//	q++;
	//}
	///**************/

	/**************/
	int p = 0;
	int q = 0;
	for (i = 0; i <= nspans; i++)
	{
		for (j = 1; j <= knotsCount[i] ; j++)
		{
			knots[p] = q;
			p++;
		}
		q++;
	}
	/**************/


	/**************/
	calBaseFunc();
	/**************/


	///**************/
	//for (i = 0; i < nspans; i++)
	//{
	//	firstKnots[i] = 0;
	//	for (j = 0; j <= i; j++)
	//	{
	//		firstKnots[i] += knotsCount[j];
	//	}
	//	firstKnots[i] -= order;
	//}
	///**************/

	/**************/
	for (i = 0; i < nspans; i++)
	{
		firstKnots[i] = 0;
		for (j = 0; j <= i; j++)
		{
			firstKnots[i] += knotsCount[j];
		}
		firstKnots[i] -= order;
	}
	/**************/


	/**************/
	for (i = 0; i < nspans; i++)
	{
		Mat spanM;
		spanM.create(order, order, CV_32FC1);
		spanM.at<float>(0, 0) = baseFuncMat[firstKnots[i]].at<float>(2, 2);	
		spanM.at<float>(1, 0) = baseFuncMat[firstKnots[i]].at<float>(2, 1);	
		spanM.at<float>(2, 0) = baseFuncMat[firstKnots[i]].at<float>(2, 0);	
		spanM.at<float>(0, 1) = baseFuncMat[firstKnots[i] + 1].at<float>(1, 2);	
		spanM.at<float>(1, 1) = baseFuncMat[firstKnots[i] + 1].at<float>(1, 1);	
		spanM.at<float>(2, 1) = baseFuncMat[firstKnots[i] + 1].at<float>(1, 0);	
		spanM.at<float>(0, 2) = baseFuncMat[firstKnots[i] + 2].at<float>(0, 2);	
		spanM.at<float>(1, 2) = baseFuncMat[firstKnots[i] + 2].at<float>(0, 1);	
		spanM.at<float>(2, 2) = baseFuncMat[firstKnots[i] + 2].at<float>(0, 0);	
		spanMat.push_back(spanM);

		//for (p = 0; p < spanM.rows; p++)
		//{
		//	for (q = 0; q < spanM.cols; q++)
		//	{
		//		cout<<spanM.at<float>(p, q)<<ends;
		//	}
		//	cout<<endl;
		//}
	}
	/**************/

	return true;
}
bool Bspline::calBaseFunc()//knots要计算好才能使用，算完以后baseFuncMat就计算好了//
{
	int i,j;
	int p, q;
	float tmp1 = 0;
	float tmp2 = 0;
	for (i = 0; i < nfuncs; i++)
	{
		Mat funcMat;
		
		funcMat.create(order, order, CV_32FC1);
		float* datapt = (float*)funcMat.data;
		if (knots[i] < knots[i + 1])
		{
			tmp1 = ((float)((knots[i + 1] - knots[i]) * (knots[i + 2] - knots[i])));
			funcMat.at<float>(0, 0) = 1.0 / tmp1;
			funcMat.at<float>(0, 1) = ((float)(-2.0 * knots[i])) / tmp1;
			funcMat.at<float>(0, 2) = ((float)(knots[i] * knots[i])) / tmp1;
		}
		else
		{
			funcMat.at<float>(0, 0) = 0;
			funcMat.at<float>(0, 1) = 0;
			funcMat.at<float>(0, 2) = 0;
		}
		if (knots[i + 1] < knots[i + 2])
		{
			tmp1 = ((float)((knots[i + 2] - knots[i]) * (knots[i + 2] - knots[i + 1])));
			tmp2 = ((float)((knots[i + 3] - knots[i + 1]) * (knots[i + 2] - knots[i + 1])));
			funcMat.at<float>(1, 0) = -1.0 / tmp1 + -1.0 / tmp2;
			funcMat.at<float>(1, 1) = ((float)(knots[i + 2] + knots[i])) / tmp1 + 
				((float)(knots[i + 3] + knots[i + 1])) / tmp2;
			funcMat.at<float>(1, 2) = ((float)(-knots[i] * knots[i + 2])) / tmp1 + 
				((float)(-knots[i + 3] * knots[i + 1])) / tmp2;
		}
		else
		{
			funcMat.at<float>(1, 0) = 0;
			funcMat.at<float>(1, 1) = 0;
			funcMat.at<float>(1, 2) = 0;
		}
		if (knots[i + 2] < knots[i + 3])
		{
			tmp1 = ((float)((knots[i + 3] - knots[i + 1]) * (knots[i + 3] - knots[i + 2])));
			funcMat.at<float>(2, 0) = 1.0 / tmp1;
			funcMat.at<float>(2, 1) = ((float)(-2.0 * knots[i + 3])) / tmp1;
			funcMat.at<float>(2, 2) = ((float)(knots[i + 3] * knots[i + 3])) / tmp1;
		}
		else
		{
			funcMat.at<float>(2, 0) = 0;
			funcMat.at<float>(2, 1) = 0;
			funcMat.at<float>(2, 2) = 0;
		}
		baseFuncMat.push_back(funcMat);
	}
	return true;
}
bool Bspline::calShowPoints()
{
	int i, j;
	float s = 0; 
	for (i = 0; i < nspans; i++)
	{
		//vector<Point> spanPoints;

		Mat spanPoints(ww, 1, CV_32FC2);
		for (j = 0; j < ww; j++)
		{
			s += interval;

			Point pt;

			float tmp1, tmp2, tmp3;

			int x1, x2, x3, y1, y2, y3;

			tmp1 = spanMat[i].at<float>(0, 0) + 
				spanMat[i].at<float>(1, 0) * s + 
				spanMat[i].at<float>(2, 0) * s * s;
			tmp2 = spanMat[i].at<float>(0, 1) +
				spanMat[i].at<float>(1, 1) * s +
				spanMat[i].at<float>(2, 1) * s * s;
			tmp3 = spanMat[i].at<float>(0, 2) +
				spanMat[i].at<float>(1, 2) * s +
				spanMat[i].at<float>(2, 2) * s * s;

			//这里要特别小心contrlPoints和knots还有firstKnots的关系！//


			x1 = *ContrlPoints.ptr<float>(knots[firstKnots[i] + 1]);
			y1 = *(ContrlPoints.ptr<float>(knots[firstKnots[i] + 1])+1);

			x2 = *ContrlPoints.ptr<float>(knots[firstKnots[i] + 2]);
			y2 = *(ContrlPoints.ptr<float>(knots[firstKnots[i] + 2])+1);

			x3 = *ContrlPoints.ptr<float>(knots[firstKnots[i] + 3]);
			y3 = *(ContrlPoints.ptr<float>(knots[firstKnots[i] + 3])+1);


			pt.x = (int)(tmp1 * x1 + tmp2 * x2 + tmp3 * x3);
			pt.y = (int)(tmp1 * y1 + tmp2 * y2 + tmp3 * y3);

			//spanPoints.push_back(pt);

			float* tmpptr = spanPoints.ptr<float>(j);
			tmpptr[0] = (float) pt.x;
			tmpptr[1] = (float) pt.y;
		}
		//showSpans.push_back(spanPoints);

		SpanPoints.push_back(spanPoints);
	}
	return true;
}


bool Bspline::show(Mat& img, 
	const Scalar& color, int thickness)
{
	int i,j;

	for(i = 0; i < SpanPoints.size(); i++)
	{
		for (j = 0; j < SpanPoints[i].rows; j++)
		{
			Point center;

			center.x = (int)*SpanPoints[i].ptr<float>(j);

			center.y = (int)*(SpanPoints[i].ptr<float>(j)+1);
			circle(img, center, 0, color, thickness);
		}
	}
	for (i = 0; i <= nspans; i++)
	{
		Point center;

		center.x = (int)*(ContrlPoints.ptr<float>(i));

		center.y = (int)*(ContrlPoints.ptr<float>(i)+1);
		circle(img, center, 0, color, thickness + 5);

	}
	return true;
}


