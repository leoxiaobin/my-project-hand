#ifndef _OPENCV_HOGFEATURES_H_
#define _OPENCV_HOGFEATURES_H_

#include "traincascade_features.h"

#define HOGF_NAME "hogFeatureParams"
struct CvHOGFeatureParams : CvFeatureParams
{
	CvHOGFeatureParams();

};

class CvHOGEvaluator : public CvFeatureEvaluator
{
public:
	virtual ~CvHOGEvaluator() {}
	virtual void init(const CvFeatureParams *_featureParams,
		int _maxSampleCount, Size _winSize );
	virtual void setImage(const Mat& img, uchar clsLabel, int idx);
	virtual float operator()(int featureIdx, int sampleIdx) const
	{ return (float)features[featureIdx].calc( sum, sampleIdx); }
	virtual void writeFeatures( FileStorage &fs, const Mat& featureMap ) const;

protected:
	virtual void generateFeatures();

	class Feature
	{
	public:
		Feature();
		Feature( int offset, int x, int y, int _cell_w, int _cell_h  ); 
		uchar calc( const vector<Mat>& _sum, size_t y ) const;
		void write( FileStorage &fs ) const;

		Rect rect;
		int p[4];
	};
	vector<Feature> features;

	vector<Mat> sum;
};

inline uchar CvHOGEvaluator::Feature::calc(const vector<Mat> &_sum, size_t y) const
{

	const double* sum0 = _sum[0].ptr<double>((int)y);
	const double* sum1 = _sum[1].ptr<double>((int)y);
	const double* sum2 = _sum[2].ptr<double>((int)y);
	const double* sum3 = _sum[3].ptr<double>((int)y);
	const double* sum4 = _sum[4].ptr<double>((int)y);
	//const double* sum5 = _sum[5].ptr<double>((int)y);
	//const double* sum6 = _sum[6].ptr<double>((int)y);
	//const double* sum7 = _sum[7].ptr<double>((int)y);
	//const double* sum8 = _sum[8].ptr<double>((int)y);

	int cval = sum2[p[0]] - sum2[p[1]] - sum2[p[2]] + sum2[p[3]];

	//return (uchar)((sum0[p[0]] - sum0[p[1]] - sum0[p[2]] + sum0[p[3]] >= cval ? 128 : 0) |   // 0
	//	(sum1[p[0]] - sum1[p[1]] - sum1[p[2]] + sum1[p[3]] >= cval ? 64 : 0) |    // 1
	//	(sum2[p[0]] - sum2[p[1]] - sum2[p[2]] + sum2[p[3]] >= cval ? 32 : 0) |    // 2
	//	(sum3[p[0]] - sum3[p[1]] - sum3[p[2]] + sum3[p[3]] >= cval ? 16 : 0) |  // 5
	//	(sum5[p[0]] - sum5[p[1]] - sum5[p[2]] + sum5[p[3]] >= cval ? 8 : 0) | // 8
	//	(sum6[p[0]] - sum6[p[1]] - sum6[p[2]] + sum6[p[3]] >= cval ? 4 : 0) |  // 7
	//	(sum7[p[0]] - sum7[p[1]] - sum7[p[2]] + sum7[p[3]] >= cval ? 2 : 0) |   // 6
	//	(sum8[p[0]] - sum8[p[1]] - sum8[p[2]] + sum8[p[3]] >= cval ? 1 : 0));     // 3

	return (uchar)((sum0[p[0]] - sum0[p[1]] - sum0[p[2]] + sum0[p[3]] >= cval ? 8 : 0) |   // 0
		(sum1[p[0]] - sum1[p[1]] - sum1[p[2]] + sum1[p[3]] >= cval ? 4 : 0) |    // 1
		(sum3[p[0]] - sum3[p[1]] - sum3[p[2]] + sum3[p[3]] >= cval ? 2 : 0) |    // 2
		(sum4[p[0]] - sum4[p[1]] - sum4[p[2]] + sum4[p[3]] >= cval ? 1 : 0)); // 5
		//(sum5[p[0]] - sum5[p[1]] - sum5[p[2]] + sum5[p[3]] >= cval ? 8 : 0) | // 8
		//(sum6[p[0]] - sum6[p[1]] - sum6[p[2]] + sum6[p[3]] >= cval ? 4 : 0) |  // 7
		//(sum7[p[0]] - sum7[p[1]] - sum7[p[2]] + sum7[p[3]] >= cval ? 2 : 0) |   // 6
		//(sum8[p[0]] - sum8[p[1]] - sum8[p[2]] + sum8[p[3]] >= cval ? 1 : 0));     // 3
}

#endif