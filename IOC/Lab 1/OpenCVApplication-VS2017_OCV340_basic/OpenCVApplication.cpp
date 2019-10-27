// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "Functions.h"

#define MAX_HUE 256
int histc_hue[MAX_HUE];


void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
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
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = MAX_PATH - val;
				dst.at<uchar>(i, j) = neg;
			}
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
		for (int i = 0; i < height; i++)
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
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
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

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i * width + j;

				dstDataPtrH[gi] = hsvDataPtr[hi] * 510 / 360;		// H = 0 .. 255
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
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
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
		Canny(grayFrame, edges, 40, 100, 3);
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
		if (c == 115) { //'s' pressed - snapp the image to a file
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

void rgbToHsv() {
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


		for (int i = 0; i < channels[0].rows; i++) {
			for (int j = 0; j < channels[0].cols; j++) {
				hH[channels[0].at<uchar>(i, j)]++;
			}
		}

		for (int i = 0; i < channels[1].rows; i++) {
			for (int j = 0; j < channels[1].cols; j++) {
				hS[channels[1].at<uchar>(i, j)]++;
			}
		}

		for (int i = 0; i < channels[2].rows; i++) {
			for (int j = 0; j < channels[2].cols; j++) {
				hV[channels[2].at<uchar>(i, j)]++;
			}
		}


		Mat dst = Mat::zeros(channels[0].rows, channels[0].cols, CV_8UC1);
		for (int i = 0; i < channels[0].rows; i++) {
			for (int j = 0; j < channels[0].cols; j++) {
				if ((channels[0].at<uchar>(i, j) * 510 / 360) < 30) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
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

//lab 3
void L3_ColorModel_Init() {
	memset(histc_hue, 0, sizeof(unsigned int)*MAX_HUE);
}

Point Pstart, Pend; // Punctele/colturile aferente selectiei ROI curente (declarateglobal)

void L3_ColorModel_Build()
{
	Mat src;
	Mat hsv;
	// Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		// Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
		GaussianBlur(src, src, Size(5, 5), 0, 0);
		//Creare fereastra pt. afisare
		namedWindow("src", 1);
		// Componenta de culoare Hue a modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		// definire pointeri la matricea (8 biti/pixeli) folosita la stocarea
		// componentei individuale H
		uchar* lpH = H.data;
		cvtColor(src, hsv, CV_BGR2HSV); // conversie RGB -> HSV
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsv.data;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				// index in matricea hsv (24 biti/pixel)
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j; // index in matricea H (8 biti/pixel)
				lpH[gi] = hsvDataPtr[hi] * 510 / 360; // lpH = 0 .. 255
			}
		}
		// Asociere functie de tratare a avenimentelor MOUSE cu ferestra curenta
		// Ultimul parametru este matricea H (valorile compunentei Hue)
		//setMouseCallback("src", &H);
		imshow("src", src);
		// Wait until user press some key
		waitKey(0);
	}
}

void CallBackFuncL3(int event, int x, int y, int flags, void* userdata)
{
	Mat* H = (Mat*)userdata;
	Rect roi; // regiunea de interes curenta (ROI)
	if (event == EVENT_LBUTTONDOWN)
	{
		// punctul de start al ROI
		Pstart.x = x;
		Pstart.y = y;
		printf("Pstart: (%d, %d) ", Pstart.x, Pstart.y);
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		// punctul de final (diametral opus) al ROI
		Pend.x = x;
		Pend.y = y;
		printf("Pend: (%d, %d) ", Pend.x, Pend.y);
		// sortare puncte dupa x si y
		//(parametrii width si height ai structurii Rect > 0)
		roi.x = min(Pstart.x, Pend.x);
		roi.y = min(Pstart.y, Pend.y);
		roi.width = abs(Pstart.x - Pend.x);
		roi.height = abs(Pstart.y - Pend.y);
		printf("Local ROI: (%d, %d), (%d, %d)\n", roi.x, roi.y, roi.x + roi.width,
			roi.y + roi.height);
		int hist_hue[MAX_HUE]; // histograma locala a lui Hue
		memset(hist_hue, 0, MAX_HUE * sizeof(int));
		// Din toata imaginea H se selecteaza o subimagine (Hroi) aferenta ROI
		Mat Hroi = (*H)(roi);
		uchar hue;
		//construieste histograma locala aferente ROI
		for (int y = 0; y < roi.height; y++)
			for (int x = 0; x < roi.width; x++)
			{
				hue = Hroi.at<uchar>(y, x);
				hist_hue[hue]++;
			}
		//acumuleaza histograma locala in cea globala/cumulativa
		for (int i = 0; i < MAX_HUE; i++)
			histc_hue[i] += hist_hue[i];
		// afiseaza histohrama locala
		showHistogram("H local histogram", hist_hue, MAX_HUE, 200, true);
		// afiseaza histohrama globala / cumulativa
		showHistogram("H global histogram", histc_hue, MAX_HUE, 200, true);
	}
}

void MyLabeling(const string& name, const Mat& src, bool output_format) {
	// dst - matrice RGB24 pt. afisarea rezultatului
	Mat dst = Mat::zeros(src.size(), CV_8UC3);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	Moments m;
	if (contours.size() > 0)
	{
		// iterate through all the top-level contours,
		// draw each connected component with its own random color
		int idx = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			const vector<Point>& c = contours[idx];

			// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=moments#moments
			m = moments(c); // calcul momente imagine
			double arie = m.m00; // aria componentei conexe idx

			if (arie > 100)
			{
				double xc = m.m10 / m.m00; // coordonata x a CM al componentei conexe idx
				double yc = m.m01 / m.m00; // coordonata y a CM al componentei conexe idx

				Scalar color(rand() & 255, rand() & 255, rand() & 255);

				// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
				if (output_format) // desenare obiecte pline ~ etichetare
					drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
				else  //desenare contur obiecte
					drawContours(dst, contours, idx, color, 1, 8, hierarchy);

				Point center(xc, yc);
				int radius = 5;

				// afisarea unor cercuri in jurul centrelor de masa
				//circle(final, center, radius,Scalar(255,255,355), 1, 8, 0);

				// afisarea unor cruci peste centrele de masa
				DrawCross(dst, center, 9, Scalar(255, 255, 355), 1);

				// https://en.wikipedia.org/wiki/Image_moment
				//calcul axa de alungire folosind momentele centarte de ordin 2
				double mc20p = m.m20 / m.m00 - xc * xc; // double mc20p = m.mu20 / m.m00;
				double mc02p = m.m02 / m.m00 - yc * yc; // double mc02p = m.mu02 / m.m00;
				double mc11p = m.m11 / m.m00 - xc * yc; // double mc11p = m.mu11 / m.m00;
				float teta = 0.5*atan2(2 * mc11p, mc20p - mc02p);
				float teta_deg = teta * 180 / PI;

				printf("\nID=%d, arie=%.0f, xc=%0.f, yc=%0.f, teta=%.2f, teta=%.2f\n", idx, arie, xc, yc, teta_deg, teta);

				int height = src.rows;
				int width = src.cols;
				float x1, x2, x3, x4, y1, y2, y3, y4, xf, yf, xs, ys, x, y;
				float slope = tan(teta);

				y1 = slope * (0 - xc) + yc;
				Point p1;
				p1.x = 0;
				p1.y = y1;
				y2 = slope * (width - 1 - xc) + yc;
				Point p2;
				p2.x = width - 1;
				p2.y = y2;
				x1 = (0 - yc) / slope + xc;
				Point p3;
				p3.x = x1;
				p3.y = 0;
				x2 = (height - 1 - yc) / slope + xc;
				Point p4;
				p4.x = x2;
				p4.y = height - 1;


				Point start, end;

				if (p3.x >= 0 && p3.y <= width - 1)
					start = p3;
				if (p4.x >= 0 && p4.y <= width - 1)
					start = p4;

				if (p1.x >= 0 && p1.y <= height - 1)
					end = p1;
				if (p2.x >= 0 && p2.y <= height - 1)
					end = p2;

				printf("slope =%.2f, %d %d, %d %d, %d %d, %d %d", slope, p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y);
				line(dst, start, end, Scalar(255, 255, 255), 5, 8);

			}
		}
	}

	imshow(name, dst);
}



int hue_mean = 16, hue_std = 5;
float k = 2.5;

void ex3_1() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		GaussianBlur(src, src, Size(5, 5), 0, 0);

		Mat hsvImg = src.clone();
		cvtColor(src, hsvImg, CV_BGR2HSV);

		int height = src.rows;
		int width = src.cols;

		Mat channels[3];
		split(hsvImg, channels);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				channels[0].at<uchar>(i, j) = channels[0].at<uchar>(i, j) * 256 / 180;
			}
		}

		int hH[256] = { 0 };


		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				hH[channels[0].at<uchar>(i, j)]++;
			}
		}

		Mat H = channels[0];
		Mat dst = H.clone();

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (hue_mean - k * hue_std >= 0 && hue_mean + k * hue_std <= 256) {
					if (H.at<uchar>(i, j) > (hue_mean - k * hue_std) && H.at<uchar>(i, j) < (hue_mean + k * hue_std))
						dst.at<uchar>(i, j) = 255;
					else
						dst.at<uchar>(i, j) = 0;
				}
			}
		}

		imshow("input image", src);
		imshow("binary image", dst);
		waitKey();
	}
}
void ex3_2() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		GaussianBlur(src, src, Size(5, 5), 0, 0);

		Mat hsvImg = src.clone();
		cvtColor(src, hsvImg, CV_BGR2HSV);

		int height = src.rows;
		int width = src.cols;

		Mat channels[3];
		split(hsvImg, channels);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				channels[0].at<uchar>(i, j) = channels[0].at<uchar>(i, j) * 256 / 180;
			}
		}

		int hH[256] = { 0 };


		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				hH[channels[0].at<uchar>(i, j)]++;
			}
		}

		Mat H = channels[0];
		Mat dst = H.clone();

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (hue_mean - k * hue_std >= 0 && hue_mean + k * hue_std <= 256) {
					if (H.at<uchar>(i, j) > (hue_mean - k * hue_std) && H.at<uchar>(i, j) < (hue_mean + k * hue_std))
						dst.at<uchar>(i, j) = 0;
					else
						dst.at<uchar>(i, j) = 255;
				}
			}
		}

		Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat element1 = getStructuringElement(MORPH_CROSS, Size(5, 5));

		dilate(dst, dst, element2, Point(-1, -1), 2);

		erode(dst, dst, element1, Point(-1, -1), 4);

		dilate(dst, dst, element2, Point(-1, -1), 2);
		imshow("binary image", dst);
		MyLabeling("contur", dst, false);

		imshow("input image", src);
		//imshow("binary image", dst);
		waitKey();
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
		printf("Lab 3\n");
		printf("12 - ex 3.1\n");
		printf("13 - ex 3.2\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
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
			L3_ColorModel_Build();
			break;
		case 12:
			ex3_1();
			break;
		case 13:
			ex3_2();
			break;
		}
	} while (op != 0);
	return 0;
}