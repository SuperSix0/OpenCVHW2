#include <io.h>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

int TL = 25;
int TH = 50;
Mat picture, GrayPicture, GaussianPicture, gradientPicture, NMSPicture, DTECPicture, BinaryzationPicture;
string path,pic;
double *angle;

int SOBEL_X[3][3] = { { -1, 0, 1 },
					  { -2, 0, 2 },
					  { -1, 0, 1 } };
int SOBEL_Y[3][3] = { { -1, -2, -1 },
					  {  0,  0,  0 },
					  {  1,  2,  1 } };

unsigned char clamp(unsigned char value)
{
	return value > 255 ? 255 : (value < 0 ? 0 : value);
}

Mat GradientFilter(Mat src, double* angle)
{
	Mat dst = Mat(src.rows, src.cols, src.type());
	int height = src.rows;
	int width = src.cols;
	int row, col;
	float x, y, g;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			x = y = 0;
			for (int subrow = -1; subrow <= 1; subrow++)
			{
				for (int subcol = -1; subcol <= 1; subcol++)
				{
					row = i + subrow;
					col = j + subcol;
					if (row < 0 || row >= height)
						row = i;
					if (col < 0 || col >= width)
						col = j;
					x += SOBEL_X[subrow + 1][subcol + 1] * src.data[row * width + col];
					y += SOBEL_Y[subrow + 1][subcol + 1] * src.data[row * width + col];
				}
			}
			g = sqrt(x * x + y * y);
			dst.data[i * width + j] = clamp((unsigned char)g);
			if (x == 0)
				angle[i * width + j] = (y > 0 ? 180.0 : 0.0);
			else if (y == 0)
				angle[i * width + j] = 90;
			else
				angle[i * width + j] = atan(x / y) + 90;
		}
	}
	return dst;
}

Mat NonMaximalSuppression(Mat src, double* angle)
{
	Mat dst = Mat(src.rows, src.cols, src.type());
	int height = src.rows;
	int width = src.cols;
	int row, col;
	int index;
	unsigned char t;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			index = i * width + j;
			t = *(dst.data + index) = *(src.data + index);
			if ((angle[index] >= 0 && angle[index] < 22.5) || (angle[index] >= 157.5 && angle[index] < 180))
			{
				row = i;
				col = (j - 1) < 0 ? j : (j - 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;

				row = i;
				col = (j + 1) >= width ? j : (j + 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;
			}
			else if (angle[index] >= 22.5 && angle[index] < 67.5)
			{
				row = (i - 1) < 0 ? i : (i - 1);
				col = (j + 1) >= width ? j : (j + 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;

				row = (i + 1) >= height ? i : (i + 1);
				col = (j - 1) < 0 ? j : (j - 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;
			}
			else if (angle[index] >= 67.5 && angle[index] < 112.5)
			{
				row = (i - 1) < 0 ? i : (i - 1);
				col = j;
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;

				row = (i + 1) >= height ? i : (i + 1);
				col = j;
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;
			}
			else if (angle[index] >= 112.5 && angle[index] < 157.5)
			{
				row = (i - 1) < 0 ? i : (i - 1);
				col = (j - 1) < 0 ? j : (j - 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;

				row = (i + 1) >= height ? i : (i + 1);
				col = (j + 1) >= width ? j : (j + 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;
			}
		}
	}
	return dst;
}

void edgeLink(Mat src, Mat dst, int row, int col, int threshold)
{
	int height = src.rows;
	int width = src.cols;
	int newrow, newcol;
	*(dst.data + row * width + col) = 255;
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			newrow = row + i;
			newcol = col + j;
			newrow = newrow < 0 ? 0 : (newrow >= height ? (height - 1) : newrow);
			newcol = newcol < 0 ? 0 : (newcol >= width ? (width - 1) : newcol);
			if (*(src.data + newrow * width + newcol) > threshold && *(dst.data + newrow * width + newcol) == 0)
			{
				edgeLink(src, dst, newrow, newcol, threshold);
				return;
			}
		}
	}
}

Mat DoubleThresholdEdgeConnection(Mat src, int lowThreshold, int highThreshold)
{
	Mat dst = Mat(src.rows, src.cols, src.type(),Scalar(0));
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (*(src.data + i * width + j) >= highThreshold && *(dst.data + i * width + j) == 0)
				edgeLink(src, dst, i, j, lowThreshold);
		}
	}
	return dst;
}

void on_Trackbar(int, void*)
{
	//cout << "Threshold value: " << endl;
	//cout << "High: " << TH << endl;
	//cout << "Low: " << TL << endl << endl;
	DTECPicture = DoubleThresholdEdgeConnection(NMSPicture, TL, TH);
	imwrite(path + "\\Ë«·§ÖµÍ¼_" + pic, DTECPicture);
	imshow("Edge Detection", DTECPicture);
}

int main()
{
	cout << "ÇëÊäÈëÍ¼Æ¬Â·¾¶£¨º¬Í¼Æ¬ÎÄ¼þÃû£©" << endl;
	cin >> path;
	
	pic = path.substr(path.find_last_of("\\") + 1);
	path = path.substr(0, path.find_last_of("\\"));
	//cout << path << endl << pic;
	//getchar();
	picture = imread(path + "\\" + pic);
	while (picture.empty())
	{
		cout << "¶ÁÈëÊ§°Ü£¬ÇëÊäÈëÕýÈ·Í¼Æ¬Â·¾¶" << endl;
		picture = imread(path + "\\" + pic);
	}

	angle = (double *)malloc(sizeof(double) * picture.rows * picture.cols);

	cvtColor(picture, GrayPicture, CV_RGB2GRAY);
	//imwrite(path + "\\»Ò¶ÈÍ¼_" + pic[i], GrayPicture);

	GaussianBlur(GrayPicture, GaussianPicture, Size(5, 5), 0, 0);
	//imwrite(path + "\\¸ßË¹ÂË²¨_" + pic[i], gaussianPicture);

	gradientPicture = GradientFilter(GaussianPicture, angle);
	imwrite(path + "\\ÌÝ¶ÈÍ¼_" + pic, gradientPicture);

	NMSPicture = NonMaximalSuppression(gradientPicture, angle);
	imwrite(path + "\\NMSÍ¼_" + pic, NMSPicture);
	free(angle);

	DTECPicture = DoubleThresholdEdgeConnection(NMSPicture, TL, TH);
	imwrite(path + "\\Ë«·§ÖµÍ¼_" + pic, DTECPicture);

	namedWindow("Edge Detection");
	imshow("Edge Detection", DTECPicture);

	createTrackbar("Low: 255", "Edge Detection", &TL, 255, on_Trackbar);
	createTrackbar("High: 255", "Edge Detection", &TH, 255, on_Trackbar);

	on_Trackbar(TL, 0);
	on_Trackbar(TH, 0);

	waitKey(0);
	return 0;
}