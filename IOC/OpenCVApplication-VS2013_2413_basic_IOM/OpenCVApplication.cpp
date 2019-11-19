// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "Functions.h"
#include <opencv2/video/tracking.hpp>
#include <queue>
#include "opencv2/objdetect/objdetect.hpp" 
#include "colorcode.h"
using namespace std;


/// Global variables
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

char* source_window = "Source image";
char* corners_window = "Corners detected";

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine
		
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		
		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i*width + j;
				
				dstDataPtrH[gi] = hsvDataPtr[hi] * 510/360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void rgbToHsv(){
	Mat rgb;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		rgb = imread(fname);
		Mat hsv = rgb.clone();
		cvtColor(rgb, hsv, COLOR_BGR2HSV);

		Mat channels[3];
		split(hsv, channels); 

		int hH[256] = { 0 };
		int hS[256] = { 0 };
		int hV[256] = { 0 };


		for (int i = 0; i < channels[0].rows; i++){
			for (int j = 0; j < channels[0].cols; j++){
				hH[channels[0].at<uchar>(i, j)]++;
			}
		}

		for (int i = 0; i < channels[1].rows; i++){
			for (int j = 0; j < channels[1].cols; j++){
				hS[channels[1].at<uchar>(i, j)]++;
			}
		}

		for (int i = 0; i < channels[2].rows; i++){
			for (int j = 0; j < channels[2].cols; j++){
				hV[channels[2].at<uchar>(i, j)]++;
			}
		}


		Mat dst = Mat::zeros(channels[0].rows, channels[0].cols, CV_8UC1);
		for (int i = 0; i < channels[0].rows; i++){
			for (int j = 0; j < channels[0].cols; j++){
				if ((channels[0].at<uchar>(i, j)*510/360) < 30){
					dst.at<uchar>(i, j) = 255;
				}
				else{
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
		
		//imshow("H", channels[0]);
		//imshow("S", channels[1]);
		//imshow("V", channels[2]);

		imshow("output", dst);

		//showHistogram("H", hH, 256, 256, true);
		//showHistogram("S", hS, 256, 256, true);
		//showHistogram("V", hV, 256, 256, true);

		waitKey();
	}
}

void segmentare(){
	Mat src;
	Mat hsv;

	//a)
	int hue_mean = 16;
	int hue_std = 5;

	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		src = imread(fname);
		GaussianBlur(src, src, Size(5, 5), 0, 0);

		Mat hsvImg = src.clone();
		cvtColor(src, hsvImg, CV_BGR2HSV);

		Mat channels[3];
		split(hsvImg, channels);

		int height = src.rows;
		int width = src.cols;

		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				channels[0].at<uchar>(i, j) = channels[0].at<uchar>(i, j) * 256 / 180;
			}
		}

		int hH[256] = { 0 };

		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				hH[channels[0].at<uchar>(i, j)]++;
			}
		}

		Mat H = channels[0];
		Mat dst = H.clone();

		int hue_mean = 16;
		int hue_std = 5;
		float k = 2.5f;

		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				if (H.at<uchar>(i, j) > (hue_mean - k*hue_std) && H.at<uchar>(i, j) < (hue_mean + k*hue_std)){
					dst.at<uchar>(i, j) = 255;
				}
				else{
					dst.at<uchar>(i, j) = 0;
				}
			}
		}

		//b)
		Mat element = getStructuringElement(MORPH_CROSS, Size(5, 5));
		erode(dst, dst, element, Point(-1, -1), 2);
		dilate(dst, dst, element, Point(-1, -1), 4);
		erode(dst, dst, element, Point(-1, -1), 2);

		//c)
		imshow("dst", dst);
		Labeling("contur", dst, false);
		
		waitKey(0);

	}
}

void callbackLab4(int event, int x, int y, int flags, void* userdata){

	Mat* H = (Mat*)userdata; 
	float hue_avg = 0; 
	Mat dst = Mat::zeros((*H).size(), CV_8U);

	if (event == EVENT_RBUTTONDOWN) 
	{ 
		printf("point %d %d\n", x, y);  
		uchar seed = (*H).at<uchar>(y, x); 
		Mat labels = Mat::zeros((*H).size(), CV_8U); 

		for (int i = -1; i <= 1; i++){ // vecinii
			for (int j = -1; j <= 1; j++){
				hue_avg += (*H).at <uchar>(y + i, x + j); 
			} 
		} 
		hue_avg = hue_avg / 9.0;

		queue <Point> que; 
		int k = 1; 
		int N = 1; 
		que.push(Point(y, x)); 
		 
		while (!que.empty()){ 
			Point oldest = que.front(); 
			que.pop(); 
			 
			int xx = oldest.x; 
			int yy = oldest.y; 
			 
			int pragT = 12; 
			 
			for (int i = -1; i <= 1; i++){
				for (int j = -1; j <= 1; j++){
					//daca coordonatele sunt in interiorul imaginii
					if ((xx + i) >= 0 && (xx + i) < labels.rows && (yy + j) >= 0 && (yy + j) < labels.cols){
						if ((abs((*H).at<uchar>(xx + i, yy + j) - hue_avg) < pragT) && 
							(labels.at<uchar>(xx + i, yy + j) == 0)){ 
							que.push(Point(xx + i, yy + j)); 
							labels.at<uchar>(xx + i, yy + j) = k; 
							hue_avg = (float)(N * hue_avg + (*H).at<uchar>(xx + i, yy + j)) / (float)(N + 1); 
							N++; 
						} 
					} 
				} 
			} 
		} 
		 
		for (int i = 0; i < dst.rows; i++){ 
			for (int j = 0; j < dst.cols; j++){ 
				if (labels.at<uchar>(i, j)){ 
					dst.at<uchar>(i, j) = 255; 
				} 
			}
		}
		imshow("Inainte de eroziune si dilatare", dst);
		 
		Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3)); 
		 
		erode(dst, dst, element1, Point(-1, -1), 2); 
		dilate(dst, dst, element1, Point(-1, -1), 4); 
		erode(dst, dst, element1, Point(-1, -1), 2); 
		 
		imshow("Dupa eroziune si dilatare", dst);
		 
	} 
} 

void regionGrowing(){
	Mat src; 
	Mat hsv; 
	Mat channels[3]; 

	char fname[MAX_PATH]; 
	while (openFileDlg(fname)) 
	{ 
		src = imread(fname); 
		int height = src.rows; 
		int width = src.cols; 
 		GaussianBlur(src, src, Size(5, 5), 0.8, 0.8);
		cvtColor(src, hsv, CV_BGR2HSV); 
		split(hsv, channels); 
		Mat H = channels[0].clone(); 
		H = H * 255 / 180;

		namedWindow("result", 1);
		setMouseCallback("result", callbackLab4, &H);
		imshow("result", src);
		waitKey(0);
	} 
} 

//lab 5
Mat mineTestColor2Gray(Mat src)
{
	char fname[MAX_PATH];
	src = imread(fname, CV_LOAD_IMAGE_COLOR);

	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1);



		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		//imshow("input image", src);
		//imshow("gray image", dst);
		waitKey();
	
	return dst;
}

//Ex1 si 2
void functieGoodFeaturesToTrack(){
	char fname[MAX_PATH];
	Mat src;
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat dst = src.clone();
		
		cvtColor(src, src, CV_BGR2GRAY);
		GaussianBlur(src, src, Size(5, 5), 0, 0);
		

		vector<Point2f> corners;
		int maxCorners = 100;
		double qualityLevel = 0.01;
		double minDistance = 10;
		int blockSize = 3;
		bool useHarrisDetector = true;
		double k = 0.04;


		// Apel functie
		goodFeaturesToTrack(src,corners,maxCorners,qualityLevel,minDistance,Mat(),blockSize,useHarrisDetector,k);
		int r = 4;
		for (int i = 0; i < corners.size(); i++){
			circle(dst, corners[i], r, Scalar(0, 255, 0), 1, 8, 0);
		}

		//imshow("image", src);
		imshow("dst", dst);
		waitKey();
	}

}
//Ex 3
void refined(){
	char fname[MAX_PATH];
	Mat src;
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat dst = src.clone();

		cvtColor(src, src, CV_BGR2GRAY);
		GaussianBlur(src, src, Size(5, 5), 0, 0);


		vector<Point2f> corners;
		int maxCorners = 100;
		double qualityLevel = 0.01;
		double minDistance = 10;
		int blockSize = 3;
		bool useHarrisDetector = true;
		double k = 0.04;


		// Apel functie
		goodFeaturesToTrack(src, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
		int r = 4;
		for (int i = 0; i < corners.size(); i++){
			circle(dst, corners[i], r, Scalar(0, 255, 0), 1, 8, 0);
		}

		Size winSize = Size(5, 5);
		Size zeroZone = Size(-1, -1);
		TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
		cornerSubPix(src, corners, winSize, zeroZone, criteria);

		FILE *f = fopen("C:/Users/Ionut/Desktop/IOC/OpenCVApplication-VS2013_2413_basic_IOM/colturi.txt", "w+");
		int i = 0;
		for (Point2f p : corners){
			i++;
			fprintf(f, "punctul %d => %f %f\n", i, p.x, p.y);
		}
		fclose(f);

		//imshow("image", src);
		imshow("dst", dst);
		waitKey();
	}

}


//ex4
/// Global variables

void cornerHarris_demo(int, void*)
{


	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result
	namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
	imshow(corners_window, dst_norm_scaled);
}

void cornerHarrisMine(){
	char fname[MAX_PATH];
	Mat src;
	while (openFileDlg(fname))
	{
		/// Load source image and convert it to gray
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		cvtColor(src, src_gray, CV_BGR2GRAY);

		/// Create a window and a trackbar
		namedWindow(source_window, CV_WINDOW_AUTOSIZE);
		createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
		imshow(source_window, src);

		cornerHarris_demo(0, 0);

		waitKey(0);
	}
}

//ex5

void functieGoodFeaturesToTrackForVideo(Mat src){

		Mat dst = src.clone();

		cvtColor(src, src, CV_BGR2GRAY);
		GaussianBlur(src, src, Size(5, 5), 0, 0);


		vector<Point2f> corners;
		int maxCorners = 100;
		double qualityLevel = 0.01;
		double minDistance = 10;
		int blockSize = 3;
		bool useHarrisDetector = true;
		double k = 0.04;


		// Apel functie
		goodFeaturesToTrack(src, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
		int r = 4;
		for (int i = 0; i < corners.size(); i++){
			circle(dst, corners[i], r, Scalar(0, 255, 0), 1, 8, 0);
		}

		//imshow("image", src);
		imshow("dst", dst);
		//waitKey();

}

void testVideoSequence2()
{
	VideoCapture cap("Videos/taxi.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		functieGoodFeaturesToTrackForVideo(frame);
		//imshow("source", frame);
		//imshow("gray", grayFrame);
		//imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}

void lab6m1()
{
	VideoCapture cap("Videos/laboratory.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
	Mat grayFrame, backgnd, diff;

	Mat edges;
	Mat frame, dst;
	char c;

	int nr = -1;
	const unsigned char Th = 15;
	const double alpha = 0.05;
	
	while (cap.read(frame))
	{
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		//imshow("source", frame);
		//imshow("gray", grayFrame);
		backgnd = Mat::zeros(grayFrame.size(), grayFrame.type());
		++nr;
		if (nr == 0){
			imshow("first frame", grayFrame);
		}
		else{
			imshow("current frame", grayFrame);
			
			absdiff(grayFrame, backgnd, diff);
			imshow("diff", diff);
			imshow("back", backgnd);
				dst = grayFrame.clone();
				for (int i = 0; i < diff.rows; i++){
					for (int j = 0; j < diff.cols; j++){
						if (diff.at<uchar>(i, j)>Th)
							dst.at<uchar>(i, j) = 255;
					}
				}
				
		}
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void segmentareObiecteLab6(){
	Mat frame, gray;
	Mat backgnd;
	Mat diff;
	Mat dst;
	char c;
	int frameNum = -1;
	int method = 0;
	printf("Alegeti o optiune pentru metoda: 1 / 2 / 3 \n");
	scanf("%d", &method);
	// method =
	// 1 - frame difference
	// 2 - running average
	// 3 - running average with selectivity
	const unsigned char Th = 25;
	const double alpha = 0.05;

	VideoCapture cap("Videos/laboratory.avi");
	if (!cap.isOpened()){
		printf("Cannot open video capture device.\n");
		exit(1);
	}
	for (;;){
		cap >> frame; // achizitie frame nou
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}

		++frameNum;
		if (frameNum == 0)
			imshow("sursa", frame); // daca este primul cadru se afiseaza doar sursa
		cvtColor(frame, gray, CV_BGR2GRAY);
		//Se initializeaza matricea / imaginea destinatie pentru fiecare frame
		//dst=gray.clone();
		// sau
		//
		dst = Mat::zeros(gray.size(), gray.type());
		const int channels_gray = gray.channels();
		//restrictionam utilizarea metodei doar pt. imagini grayscale cu un canal (8 bit / pixel)
		if (channels_gray > 1)
			return;
		if (frameNum > 0) // daca nu este primul cadru
		{

			absdiff(gray, backgnd, diff);
			double t = (double)getTickCount();

			if (method == 1){
				backgnd = gray.clone();
				for (int i = 0; i < diff.rows; i++){
					for (int j = 0; j < diff.cols; j++){
						if (diff.at<uchar>(i, j) > Th){
							dst.at<uchar>(i, j) = 255;
						}
					}
				}
			}

			if (method == 2){
				addWeighted(gray, alpha, backgnd, 1.0 - alpha, 0, backgnd);
				for (int i = 0; i < diff.rows; i++){
					for (int j = 0; j < diff.cols; j++){
						if (diff.at<uchar>(i, j) > Th){
							dst.at<uchar>(i, j) = 255;
						}
					}
				}
			}

			if (method == 3){
				for (int i = 0; i < diff.rows; i++){
					for (int j = 0; j < diff.cols; j++){
						if (diff.at<uchar>(i, j) > Th){
							dst.at<uchar>(i, j) = 255;
						}
						else{
							backgnd.at<uchar>(i, j) = alpha*gray.at<uchar>(i, j) + (1.0 - alpha)*backgnd.at<uchar>(i, j);
						}
					}
				}
			}

			Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
			erode(dst, dst, element, Point(-1, -1), 2);
			dilate(dst, dst, element, Point(-1, -1), 2);

			// Get the current time again and compute the time difference [s]
			t = ((double)getTickCount() - t) / getTickFrequency();
			// Print (in the console window) the processing time in [ms]
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);

			imshow("sursa", frame); // show source
			imshow("dest", dst); // show destination

		}
		else // daca este primul cadru, modelul de fundal este chiar el
			backgnd = gray.clone();
		// Conditia de avansare/terminare in cilului for(;;) de procesare
		c = cvWaitKey(0); // press any key to advance between frames
		//for continous play use cvWaitKey( delay > 0)
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - playback finished\n");
			break; //ESC pressed
		}
	}
}

//lab 7

void lab7(){
	Mat frame, crnt;
	Mat prev;
	Mat dst;
	Mat flow;

	int maxCorners = 100;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blocksize = 3;
	bool useHarrisDetector = true;
	double k = 0.04;

	vector<Point2f> prev_pts;
	vector<Point2f> crnt_pts;
	vector<uchar> status;

	vector<float> error;

	Size winSize = Size(21, 21);

	int maxLevel = 3;

	TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03);
	int flags = 0;
	double minEigThreshold = 1e-4;

	char c;
	char fname[MAX_PATH];
	openFileDlg(fname);
	VideoCapture cap(fname);

	printf("1-HS\n2-LK\n3-PyramidLK\n");
	int method;
	scanf("%d", &method);

	int frameNum = -1;

	for (;;){
		cap >> frame;

		if (frame.empty()){
			printf("End of video file\n");
			break;
		}

		++frameNum;

		cvtColor(frame, crnt, CV_BGR2GRAY);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);

		if (frameNum > 0){

			switch (method){
			case 1:
				calcOpticalFlowHS(prev, crnt, 0, 0.1, TermCriteria(TermCriteria::MAX_ITER, 16, 0), flow);
				showFlow("HS", prev, flow, 1, 4, true, true, false);
				break;
			case 2:
				calcOpticalFlowLK(prev, crnt, Size(15, 15), flow);
				showFlow("LK", prev, flow, 1, 4, true, true, false);
				break;
			case 3:
				goodFeaturesToTrack(prev, prev_pts, maxCorners, qualityLevel, minDistance, Mat(), blocksize, useHarrisDetector, k);
				calcOpticalFlowPyrLK(prev, crnt, prev_pts, crnt_pts, status, error, winSize, maxLevel, criteria);
				showFlowSparse("PyrLK", prev, prev_pts, crnt_pts, status, error, 2, true, true, true);
				break;
			}
		}

		c = cvWaitKey(0);

		prev = crnt.clone();



		if (c == 27){
			printf("ESC pressed");
			break;
		}
	}
}

void analizaMiscariiPeBazaFluxuluiOpticDensFarneback(){
	makeColorwheel();
	make_HSI2RGB_LUT();
	Mat frame, crnt;
	Mat prev;
	Mat dst;
	Mat flow;

	char c;
	char fname[MAX_PATH];
	openFileDlg(fname);
	VideoCapture cap(fname);

	int frameNum = -1;

	for (;;){
		cap >> frame;

		if (frame.empty()){
			printf("End of video file\n");
			break;
		}

		++frameNum;

		cvtColor(frame, crnt, CV_BGR2GRAY);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);

		int winSize = 11;

		if (frameNum > 0){
			double t = (double)getTickCount();
			calcOpticalFlowFarneback(prev, crnt, flow, 0.5, 3, winSize, 10, 6, 1.5, 0);
			showFlowDense("Farneback", prev, flow, 1.0, true);
			//showFlow("Farneback", prev, flow, 1, 4, true, true, false);
			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);
		}

		c = cvWaitKey(0);

		prev = crnt.clone();



		if (c == 27){
			printf("ESC pressed");
			break;
		}
	}
}

void analizaMiscariiPeBazaFluxuluiOpticDensMiddleburry(){
	makeColorwheel();
	make_HSI2RGB_LUT();
	Mat frame, crnt;
	Mat prev;
	Mat dst;
	Mat flow;

	char c;
	char fname[MAX_PATH];
	openFileDlg(fname);
	VideoCapture cap(fname);

	int frameNum = -1;

	for (;;){
		cap >> frame;

		if (frame.empty()){
			printf("End of video file\n");
			break;
		}

		++frameNum;

		cvtColor(frame, crnt, CV_BGR2GRAY);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		
		int winSize = 11;
		float minVel = 2.5;

		if (frameNum > 0){
			double t = (double)getTickCount();
			calcOpticalFlowFarneback(prev, crnt, flow, 0.5, 3, winSize, 10, 6, 1.5, 0);
			showFlowDense("Middleburry", prev, flow, minVel, true);
			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);

		}

		c = cvWaitKey(0);

		prev = crnt.clone();



		if (c == 27){
			printf("ESC pressed");
			break;
		}
	}
}

void calcululHistogrameiDirectiilorVectorilorDeMiscare(){
	Mat frame, crnt;
	Mat prev;
	Mat dst;
	Mat flow;

	char c;
	char fname[MAX_PATH];
	openFileDlg(fname);
	VideoCapture cap(fname);

	int frameNum = -1;

	for (;;){
		cap >> frame;

		if (frame.empty()){
			printf("End of video file\n");
			break;
		}

		++frameNum;

		cvtColor(frame, crnt, CV_BGR2GRAY);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		makeColorwheel();
		make_HSI2RGB_LUT();
		int winSize = 11;
		float minVel = 1;
		int hist_cols = 360;
		int *hist_dir;

		if (frameNum > 0){
			double t = (double)getTickCount();

			calcOpticalFlowFarneback(prev, crnt, flow, 0.5, 3, winSize, 10, 6, 1.5, 0);

			hist_dir = new int[hist_cols];

			for (int i = 0; i < hist_cols; i++){
				hist_dir[i] = 0;
			}

			for (int i = 0; i < flow.rows; i++){
				for (int j = 0; j < flow.cols; j++){
					Point2f f = flow.at<Point2f>(i, j);
					float dir_rad = PI + atan2(-f.y, -f.x);
					int dir_deg = dir_rad * 180 / PI;
					float mod = sqrt(f.x*f.x + f.y*f.y);
					if (mod >= minVel)
						hist_dir[dir_deg]++;
				}
			}
			showHistogram("Hist", hist_dir, hist_cols, 200, true);
			showHistogramDir("HistDir", hist_dir, hist_cols, 200, true);

			showFlowDense("Middleburry", prev, flow, minVel, true);

			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);

		}

		c = cvWaitKey(0);

		prev = crnt.clone();



		if (c == 27){
			printf("ESC pressed");
			break;
		}
	}
}

//lab 8 (9) 

extern CascadeClassifier face_cascade;
extern CascadeClassifier eyes_cascade;
extern CascadeClassifier nose_cascade;
extern CascadeClassifier mouth_cascade;




void faceDetectionAll(){

	Mat src, dst;
	vector<Rect> faces;
	//String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String nose_cascade_name = "haarcascade_mcs_nose.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
	// Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}
	if (!nose_cascade.load(nose_cascade_name))
	{
		printf("Error loading nose cascades !\n");
		return;
	}
	if (!mouth_cascade.load(mouth_cascade_name))
	{
		printf("Error loading mouth cascades !\n");
		return;
	}
	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5; // conform proprietatilor antropomorfice ale

		FaceDetectandDisplayAll("Dst", dst, minFaceSize, minEyeSize, faces);
		waitKey(0);
	}

}

void faceDetectionVideo(){
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return;

	String face_cascade_name = "lbpcascade_frontalface.xml";

	// Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	Mat  frame_gray;
	namedWindow("camera", 1);
	for (;;)
	{
		std::vector<Rect> faces;
		Mat frame;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		equalizeHist(frame_gray, frame_gray);
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5;
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minFaceSize, minFaceSize));

		for (int i = 0; i < faces.size(); i++)
		{
			// get the center of the face
			Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			// draw circle around the face
			//ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0,
			//360, Scalar(255, 0, 255), 4, 8, 0);
			Point stanga_sus(faces[i].x, faces[i].y);
			Point dreapta_jos(faces[i].x + faces[i].height, faces[i].y + faces[i].width);

			rectangle(frame, stanga_sus, dreapta_jos, Scalar(255, 0, 255), 1, 8, 0);


		}

		imshow("camera", frame);
		if (waitKey(30) >= 0)
			break;
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Lab1\n");
		printf(" 11 - Segmentaren\n");
		printf(" 12 - Region Growing\n");
		printf(" 13 - functieGoodFeaturesToTrack\n");
		printf(" 14 - refined\n");
		printf(" 15 - cornerHarris_demo\n");
		printf(" 16 - test video seq\n");
		printf(" 17 - difference\n");
		printf(" 18 - lab6 segmentare obiecte\n");
		printf(" 19 - lab7\n");
		printf(" 20 - lab8.1\n");
		printf(" 21 - lab8.2\n");
		printf(" 22 - lab8 Farneback\n");
		printf(" 23 - lab9 Facedetection\n");
		printf(" 24 - lab9 Facedetection video\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				rgbToHsv();
				break;
			case 11:
				segmentare();
				break;
			case 12:
				regionGrowing();
				break;
			case 13:
				functieGoodFeaturesToTrack();
				break;
			case 14:
				refined();
				break;
			case 15:
				cornerHarrisMine();
				break;
			case 16:
				testVideoSequence2();
				break;
			case 17:
				lab6m1();
				break;
			case 18:
				segmentareObiecteLab6();
				break;
			case 19:
				lab7();
				break;
			case 20:
				analizaMiscariiPeBazaFluxuluiOpticDensFarneback();
				break;
			case 21:
				analizaMiscariiPeBazaFluxuluiOpticDensMiddleburry();
				break;
			case 22:
				calcululHistogrameiDirectiilorVectorilorDeMiscare();
				break;
			case 23:
				faceDetectionAll();
				break;
			case 24:
				faceDetectionVideo();
				break;
		}
	}
	while (op!=0);
	return 0;
}