#include <io.h>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

void getAllFiles(string path, vector<string>& pic)
{
	long File = 0;
	struct _finddata_t fileinfo;
	string p = path;
	if ((File = _findfirst((p + "\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (strstr(fileinfo.name, ".bmp") != NULL
				|| strstr(fileinfo.name, ".dib") != NULL
				|| strstr(fileinfo.name, ".jpeg") != NULL
				|| strstr(fileinfo.name, ".jpg") != NULL
				|| strstr(fileinfo.name, ".jpe") != NULL
				|| strstr(fileinfo.name, ".jp2") != NULL
				|| strstr(fileinfo.name, ".png") != NULL
				|| strstr(fileinfo.name, ".pbm") != NULL
				|| strstr(fileinfo.name, ".pgm") != NULL
				|| strstr(fileinfo.name, ".ppm") != NULL
				|| strstr(fileinfo.name, ".sr") != NULL
				|| strstr(fileinfo.name, ".ras") != NULL
				|| strstr(fileinfo.name, ".tiff") != NULL
				|| strstr(fileinfo.name, ".tif") != NULL)		//寻找符合格式的图片文件
				pic.push_back(fileinfo.name);
		} while (_findnext(File, &fileinfo) == 0);
		_findclose(File);
	}
}

int SOBEL_X[3][3] ={{ -1, 0, 1 },
					{ -2, 0, 2 },
					{ -1, 0, 1 } };
int SOBEL_Y[3][3] ={{ -1, -2, -1 },
					{  0,  0,  0 },
					{  1,  2,  1 } };

unsigned char clamp(float value)
{
	return value > 255 ? 255 : (value < 0 ? 0 : value);
}

void GradientFilter(Mat src, Mat dst, float* angle)
{
	dst = Mat(src.rows, src.cols, src.type());
	int height = src.rows;
	int width = src.cols;
	int row, col;
	float x,y,g;
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
			dst.data[i * width + j] = clamp(g);
			if(x == 0)
				angle[i * width + j] = (y > 0 ? 180 : 0);
			else if (y == 0)
				angle[i * width + j] = 90;
			else
				angle[i * width + j] = atan(x / y) + 90;
		}
	}
}

void NonMaximalSuppression(Mat src, Mat dst, float* angle)
{
	dst = Mat(src.rows, src.cols, src.type());
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
				col = (j - 1) < 0 ? j : (j -1);
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

				row = (i + 1) >=height ? i : (i + 1);
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
}

void edgeLink(Mat src, Mat dst, int row, int col, float threshold)
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
			if (*(src.data + newrow * width + newcol) >= threshold && *(dst.data + newrow * width + newcol) == 0)
			{
				edgeLink(src, dst, newrow, newcol, threshold);
				return;
			}
		}
	}
}

void DoubleThresholdEdgeConnection(Mat src, Mat dst, float lowThreshold,float highThreshold)
{
	dst = Mat(src.rows, src.cols, src.type(),Scalar(0));
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
}

int main(int argc, char** argv)
{
	string path;
	vector <string> pic;
	Mat picture, GrayPicture, GaussianPicture, gradientPicture, NMSPicture, DTECPicture, BinaryzationPicture;
	float *angle;
	int lowThreshold, highThreshold;
	cout << "请输入TL TH：" << endl;
	cin >> lowThreshold >> highThreshold;
	for (int i = 1; i < argc; i++)
	{
		path = *(argv + i);
		pic.clear();
		getAllFiles(path, pic);
		for (unsigned int i = 0; i < pic.size(); i++)
		{
			picture = imread(path + "\\" + pic[i]);
			angle = (float *)malloc(sizeof(float) * picture.rows * picture.cols);
			
			cvtColor(picture, GrayPicture, CV_RGB2GRAY);
			//imwrite(path + "\\灰度图_" + pic[i], GrayPicture);
			
			GaussianBlur(GrayPicture, GaussianPicture, Size(5, 5), 0, 0);
			//imwrite(path + "\\高斯滤波_" + pic[i], gaussianPicture);
			
			GradientFilter(GaussianPicture, gradientPicture, angle);
			imwrite(path + "\\梯度图_" + pic[i], gradientPicture);
			
			NonMaximalSuppression(gradientPicture, NMSPicture, angle);
			imwrite(path + "\\NMS图_" + pic[i], NMSPicture);
			
			DoubleThresholdEdgeConnection(NMSPicture, DTECPicture, lowThreshold, highThreshold);
			imwrite(path + "\\双阀值图_" + pic[i], DTECPicture);

			free(angle);
		}
	}
	return 0;
}