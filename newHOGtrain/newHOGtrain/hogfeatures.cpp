#include "hogfeatures.h"
#include "cascadeclassifier.h"

CvHOGFeatureParams::CvHOGFeatureParams()
{
	maxCatCount = 16;
	name = HOGF_NAME;
}

void CvHOGEvaluator::init(const CvFeatureParams *_featureParams, int _maxSampleCount, Size _winSize)
{
	CV_Assert( _maxSampleCount > 0);
	for (int i = 0; i<5; i++)
	{
		Mat tmp = Mat::zeros((int)_maxSampleCount, (_winSize.width + 1) * (_winSize.height + 1), CV_64FC1);
		sum.push_back(tmp);
	}
	//sum.create((int)_maxSampleCount, (_winSize.width + 1) * (_winSize.height + 1), CV_32SC1);
	CvFeatureEvaluator::init( _featureParams, _maxSampleCount, _winSize );
}



void CvHOGEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
	CV_DbgAssert( !sum.empty() );
	CvFeatureEvaluator::setImage( img, clsLabel, idx );
	
	CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );

	int imgWidth = img.cols;
	int imgHeight = img.rows;

	Size wholeSize;
	Point roiofs;
	img.locateROI(wholeSize, roiofs);

	Mat bins = Mat::zeros(5,imgWidth*imgHeight,CV_32FC1);

	int i, x, y;
	int cn = img.channels();

	Mat_<float> _lut(1, 256);
	const float* lut = &_lut(0,0);

	for( i = 0; i < 256; i++ )
		_lut(0,i) = std::sqrt((float)i);

	AutoBuffer<int> mapbuf(imgWidth + imgHeight + 4);
	int* xmap = (int*)mapbuf + 1;
	int* ymap = xmap + imgWidth + 2;

	const int borderType = (int)BORDER_REFLECT_101;

	for( x = -1; x < imgWidth + 1; x++ )
		xmap[x] = borderInterpolate(x + roiofs.x,
		wholeSize.width, borderType);
	for( y = -1; y < imgHeight + 1; y++ )
		ymap[y] = borderInterpolate(y + roiofs.y,
		wholeSize.height, borderType);

	// x- & y- derivatives for the whole row
	//int width = imgWidth;
	AutoBuffer<float> _dbuf(imgWidth*4);
	float* dbuf = _dbuf;
	Mat Dx(1, imgWidth, CV_32F, dbuf);
	Mat Dy(1, imgWidth, CV_32F, dbuf + imgWidth);
	Mat Mag(1, imgWidth, CV_32F, dbuf + imgWidth*2);
	Mat Angle(1, imgWidth, CV_32F, dbuf + imgWidth*3);

	int _nbins = 5;
	float angleScale = (float)(_nbins/CV_PI);

	for( y = 0; y < imgHeight; y++ )
	{
		const uchar* imgPtr = img.data + img.step*ymap[y];
		const uchar* prevPtr = img.data + img.step*ymap[y-1];
		const uchar* nextPtr = img.data + img.step*ymap[y+1];

		if( cn == 1 )
		{
			for( x = 0; x < imgWidth; x++ )
			{
				int x1 = xmap[x];
				dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
				dbuf[imgWidth + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
			}
		}
		else
		{
			for( x = 0; x < imgWidth; x++ )
			{
				int x1 = xmap[x]*3;
				const uchar* p2 = imgPtr + xmap[x+1]*3;
				const uchar* p0 = imgPtr + xmap[x-1]*3;
				float dx0, dy0, dx, dy, mag0, mag;

				dx0 = lut[p2[2]] - lut[p0[2]];
				dy0 = lut[nextPtr[x1+2]] - lut[prevPtr[x1+2]];
				mag0 = dx0*dx0 + dy0*dy0;

				dx = lut[p2[1]] - lut[p0[1]];
				dy = lut[nextPtr[x1+1]] - lut[prevPtr[x1+1]];
				mag = dx*dx + dy*dy;

				if( mag0 < mag )
				{
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dx = lut[p2[0]] - lut[p0[0]];
				dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];
				mag = dx*dx + dy*dy;

				if( mag0 < mag )
				{
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dbuf[x] = dx0;
				dbuf[x+imgWidth] = dy0;
			}
		}

		cartToPolar( Dx, Dy, Mag, Angle, false );

		for( x = 0; x < imgWidth; x++ )
		{
			float mag = dbuf[x+imgWidth*2], angle = dbuf[x+imgWidth*3]*angleScale - 0.5f;
			int hidx = cvFloor(angle);
			angle -= hidx;
			if( hidx < 0 )
				hidx += _nbins;
			else if( hidx >= _nbins )
				hidx -= _nbins;
			assert( (unsigned)hidx < (unsigned)_nbins );

			float* tmpPtr = (float*)bins.ptr(hidx);
			tmpPtr[y*imgHeight+x] = mag*(1.f-angle);

			hidx++;
			hidx &= hidx < _nbins ? -1 : 0;

			tmpPtr = (float*)bins.ptr(hidx);
			tmpPtr[y*imgHeight+x] = mag*angle;
		}
	}

	//vector<Mat> innSum;

	Mat innSum0((imgHeight+1),(imgWidth+1),CV_64FC1,(double*)sum[0].ptr((int)idx));
	Mat innSum1((imgHeight+1),(imgWidth+1),CV_64FC1,(double*)sum[1].ptr((int)idx));
	Mat innSum2((imgHeight+1),(imgWidth+1),CV_64FC1,(double*)sum[2].ptr((int)idx));
	Mat innSum3((imgHeight+1),(imgWidth+1),CV_64FC1,(double*)sum[3].ptr((int)idx));
	Mat innSum4((imgHeight+1),(imgWidth+1),CV_64FC1,(double*)sum[4].ptr((int)idx));
	//Mat innSum5((imgHeight+1),(imgWidth+1),CV_64FC1,(double*)sum[5].ptr((int)idx));
	//Mat innSum6((imgHeight+1),(imgWidth+1),CV_64FC1,(double*)sum[6].ptr((int)idx));
	//Mat innSum7((imgHeight+1),(imgWidth+1),CV_64FC1,(double*)sum[7].ptr((int)idx));
	//Mat innSum8((imgHeight+1),(imgWidth+1),CV_64FC1,(double*)sum[8].ptr((int)idx));

	//integralBins.create(9,(imgWidth+1)*(imgHeight+1), CV_64FC1);

	Mat tmpBins0(imgHeight,imgWidth,CV_32FC1,(float*)bins.ptr(0));
	Mat tmpBins1(imgHeight,imgWidth,CV_32FC1,(float*)bins.ptr(1));
	Mat tmpBins2(imgHeight,imgWidth,CV_32FC1,(float*)bins.ptr(2));
	Mat tmpBins3(imgHeight,imgWidth,CV_32FC1,(float*)bins.ptr(3));
	Mat tmpBins4(imgHeight,imgWidth,CV_32FC1,(float*)bins.ptr(4));
	//Mat tmpBins5(imgHeight,imgWidth,CV_32FC1,(float*)bins.ptr(5));
	//Mat tmpBins6(imgHeight,imgWidth,CV_32FC1,(float*)bins.ptr(6));
	//Mat tmpBins7(imgHeight,imgWidth,CV_32FC1,(float*)bins.ptr(7));
	//Mat tmpBins8(imgHeight,imgWidth,CV_32FC1,(float*)bins.ptr(8));

	integral(tmpBins0,innSum0);
	integral(tmpBins1,innSum1);
	integral(tmpBins2,innSum2);
	integral(tmpBins3,innSum3);
	integral(tmpBins4,innSum4);
	//integral(tmpBins5,innSum5);
	//integral(tmpBins6,innSum6);
	//integral(tmpBins7,innSum7);
	//integral(tmpBins8,innSum8);

}

void CvHOGEvaluator::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
{
	_writeFeatures( features, fs, featureMap );
}

void CvHOGEvaluator::generateFeatures()
{
	int offset = winSize.width + 1;
	for( int x = 0; x < winSize.width; x++ )
		for( int y = 0; y < winSize.height; y++ )
			for( int w = 1; w <= winSize.width; w++ )
				for( int h = 1; h <= winSize.height; h++ )
					if ( (x+w <= winSize.width) && (y+h <= winSize.height) )
						features.push_back( Feature(offset, x, y, w, h ) );
	numFeatures = (int)features.size();
}

CvHOGEvaluator::Feature::Feature()
{
	rect = cvRect(0, 0, 0, 0);
}

CvHOGEvaluator::Feature::Feature( int offset, int x, int y, int _cellWidth, int _cellHeight )
{
	Rect tr = rect = cvRect(x, y, _cellWidth, _cellHeight);
	CV_SUM_OFFSETS( p[0], p[1], p[2], p[3], tr, offset )
}

void CvHOGEvaluator::Feature::write(FileStorage &fs) const
{
	fs << CC_RECT << "[:" << rect.x << rect.y << rect.width << rect.height << "]";
}
