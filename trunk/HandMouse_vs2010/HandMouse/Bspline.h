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
	   ������������Ժ�spanMat��span matrix���ͼ������ 
	   spanMat��nspans������
	   ����Ĵ�С����3*3����ΪorderΪ3��,
	   �������֯��ʽΪ��
	   func1 func2 func3
	   a00   a01   a02  1    
	   a10   a11   a12  s
	   a20   a21   a22  s*s
	   (���һ��span��Ҫ��������
	   a00, a10, a20�����������ֱ��ʾ�����һ��span�ĵ�1�����̵ĳ����s�s*s���ϵ��)
	*/
	/************************************************************************/
	bool calSpanMatrices();


	/************************************************************************/
	/* 
	   ������������Ժ�baseFuncMat�ͼ������ 
	   baseFuncMat��nfuncs��nfuncs = nknots - order��������
	   ����Ĵ�С����3*3����ΪorderΪ3��,
	   �������֯��ʽΪ��
	   s*s    s    1
	   a00   a01  a02  0�Σ�a00, a01, a02�����������ֱ��ʾ�˻����̵�0�ε�s*s��, s��ͳ������ϵ����
	   a10   a11  a12  1��
	   a20   a21  a22  2��
	*/
	/************************************************************************/
	

	/************************************************************************/
	/*
	���������Ҫ�����к���calSpanMatrices֮����ܹ�ʹ�ã�
	�����꺯��calSpanMatrices�Ժ󣬴�B������һ�в�����������ˣ�
	Ȼ������һ��ͼƬimg������img������ʾ�������Point* contrlPoints��ʾ��B�����ˡ�
	*/
	/************************************************************************/
	bool show(Mat& img,
		const Scalar& color, int thickness=1);

	bool calShowPoints();
	
	vector<Mat> getSpanPoints() {return SpanPoints;}

private:
	int nspans;
	int nknots;//knots�ĸ���//
	int * knots;//ÿ��knots��ֵ//
	int * firstKnots;// ��¼ÿ��span�ĵ�һ��knot��λ��//
	int order;//3
	int nfuncs;//�������ĸ��� = nknots - order//
	float interval;//B�����Ĳ������//
	int ww;
	const int* knotsCount;//ÿ�����ϵ�knots����//
	vector<Mat> baseFuncMat;//�������󷽳�//
	vector<Mat> spanMat;//span matrix//
	//Point* contrlPoints;//���Ƶ�//
	Mat KnotsCount;
	Mat ContrlPoints;
	vector< vector<Point> > showSpans;//B������ʾ��//

	vector<Mat> SpanPoints;

	bool calBaseFunc();

};
#endif