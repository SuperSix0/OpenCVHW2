# OpenCVHW2
Canny Edge 检测算法
使用Visual Studio 2017 + OpenCV实现Canny Edge 检测算法

环境配置：
从https://opencv.org/releases.html下载OpenCV安装包后解压到C盘根目录
新建Visual C++空项目
	打开项目属性
		平台选择x64
		在VC++目录-包含目录中新增包含目录：
			C:\opencv\build\include
			C:\opencv\build\include\opencv
			C:\opencv\build\include\opencv2
		在VC++目录-库目录中新增库目录：
			C:\opencv\build\x64\vc14\lib
		在链接器-输入-附加依赖项中新增附加依赖项：
			opencv_world331.lib
			opencv_world331d.lib
		在环境变量path中增加动态链接库.dll文件的目录：
			C:\opencv\build\x64\vc14\bin

灰度处理和高斯模糊使用OpenCV函数实现
图像梯度、非极大抑制、双阀值自己实现

程序使用方法：
在VS2017中生成解决方案，然后使用cmd命令行进入OpenCVExp2.exe所在的目录
输入OpenCVExp2 + 图片的全路径，如：
OpenCVExp2 C:\Sourcematerial\***.jpg
即可在图片所在目录生成梯度图、NMS图、双阀值图
生成一个窗口显示双阀值图，拖动滑块可控制TL、TH大小