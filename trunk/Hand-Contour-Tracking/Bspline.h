#ifndef BSPLINE_H
#define BSPLINE_H
#include "opencv.h"
class Bspline
{
public:
	Bspline(const int _nspans, const int _nknots, const int _order,  const int* _knotsCount, Mat& _controlPoints,
		 const float _interval);
	~Bspline();


	/************************************************************************/
	/* 
	   运行这个函数以后spanMat（span matrix）就计算好了 
	   spanMat有nspans个矩阵，
	   矩阵的大小都是3*3（因为order为3）,
	   矩阵的组织方式为：
	   func1 func2 func3
	   a00   a01   a02  1    
	   a10   a11   a12  s
	   a20   a21   a22  s*s
	   (组成一个span需要三个方程
	   a00, a10, a20这三个参数分别表示组成这一个span的第1个方程的常数项，s项，s*s项的系数)
	*/
	/************************************************************************/
	bool calSpanMatrices();


	/************************************************************************/
	/* 
	   运行这个函数以后baseFuncMat就计算好了 
	   baseFuncMat有nfuncs（nfuncs = nknots - order）个矩阵，
	   矩阵的大小都是3*3（因为order为3）,
	   矩阵的组织方式为：
	   s*s    s    1
	   a00   a01  a02  0段（a00, a01, a02这三个数，分别表示此基方程第0段的s*s项, s项和常数项的系数）
	   a10   a11  a12  1段
	   a20   a21  a22  2段
	*/
	/************************************************************************/
	

	/************************************************************************/
	/*
	这个函数需要在运行函数calSpanMatrices之后才能够使用，
	运行完函数calSpanMatrices以后，此B样条的一切参数都计算好了，
	然后输入一张图片img就能在img上面显示由输入的Point* contrlPoints表示的B样条了。
	*/
	/************************************************************************/
	bool show(Mat& img,
		const Scalar& color, int thickness=1);

	bool calShowPoints();
	
	vector<Mat> getSpanPoints() {return SpanPoints;}

private:
	int nspans;
	int nknots;//knots的个数//
	int * knots;//每个knots的值//
	int * firstKnots;// 记录每个span的第一个knot的位置//
	int order;//3
	int nfuncs;//基函数的个数 = nknots - order//
	float interval;//B样条的采样间隔//
	int ww;
	const int* knotsCount;//每个点上的knots个数//
	vector<Mat> baseFuncMat;//基础矩阵方程//
	vector<Mat> spanMat;//span matrix//
	//Point* contrlPoints;//控制点//
	Mat KnotsCount;
	Mat ContrlPoints;
	vector< vector<Point> > showSpans;//B样条显示点//

	vector<Mat> SpanPoints;

	bool calBaseFunc();

};
#endif