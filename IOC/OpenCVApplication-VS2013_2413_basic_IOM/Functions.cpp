#include "stdafx.h"
#include "common.h"
#include "Functions.h"

//unsigned char HSI2RGB[360][3];
float HSI2RGB[360][3]; // LUT to convert HSI values to RGB

void make_HSI2RGB_LUT()
{
unsigned char I = 75;
float S = 0.8;
int H;
const float pi = 3.141592653f;
float d2r;
int R = 0;
int G = 1;
int B = 2;

for (H = 0; H<360; H++)
{
	d2r = pi/180; // deg 2 radians scaling

	if (H == 0) 
	{
		HSI2RGB[H][R] = I + 2*I*S;
		HSI2RGB[H][G] = I - I*S;
		HSI2RGB[H][B] = I - I*S;
	}
	
	if ( 0 < H && H < 120)
	{
		HSI2RGB[H][R] = I + I*S*cos(H*d2r)/cos((60-H)*d2r);
		HSI2RGB[H][G] = I + I*S*(1-cos(H*d2r)/cos((60-H)*d2r));
		HSI2RGB[H][B] = I - I*S;
	}

	if (H == 120)
	{
		HSI2RGB[H][R] = I - I*S;
		HSI2RGB[H][G] = I + 2*I*S;
		HSI2RGB[H][B] = I - I*S;
	}

	if ( 120 < H && H < 240)
	{
		HSI2RGB[H][R] = I - I*S;  
		HSI2RGB[H][G] = I + I*S*cos((H-120)*d2r)/cos((180-H)*d2r);
		HSI2RGB[H][B] = I + I*S*(1-cos((H-120)*d2r))/cos((180-H)*d2r);
	}

	if (H == 240)
	{
		HSI2RGB[H][R] = I - I*S;
		HSI2RGB[H][G] = I - I*S;
		HSI2RGB[H][B] = I + 2*I*S;
	}


	if ( 240 < H && H < 360)
	{
		HSI2RGB[H][R] = I + I*S*(1-cos((H-240)*d2r))/cos((300-H)*d2r);   
		HSI2RGB[H][G] = I - I*S;
		HSI2RGB[H][B] = I + I*S*cos((H-240)*d2r)/cos((300-H)*d2r);
	}

 }

}


/* wrapper functions for legacy optical flow methods
------------------------------------------------------------------------------------ */

Mat convert2flow(const Mat& velx, const Mat& vely)
// converts the optical flow vectors velx and vely in a matrix flow 
//in wich each element encodes the 2 components of the optical flow vor each pixel
{
	Mat flow(velx.size(), CV_32FC2);
	for (int y = 0; y < flow.rows; ++y)
		for (int x = 0; x < flow.cols; ++x)
			flow.at<Point2f>(y, x) = Point2f(velx.at<float>(y, x), vely.at<float>(y, x));
	return flow;
}


/*------------------------------------------------
Wrapper function that calls the  legacy implementation of the Lukas-Kanade opical flow method (cvCalcOpticalFlowLK)
Input:
prev - Previous image, 8-bit, single-channel
crnt - crntent image, 8-bit, single-channel
win_size – Size of the averaging window used for grouping pixels (Size.x, Size.y <=15)
Output:
flow- matrix containing the optical flow vectors/pixel
Call example:
calcOpticalFlowLK( prev, crnt, Size(15, 15), flow);
-------------------------------------------------------------*/
void calcOpticalFlowLK(const Mat& prev, const Mat& crnt, Size winSize, Mat& flow)
{
	Mat velx(prev.size(), CV_32F), vely(prev.size(), CV_32F);
	CvMat cvvelx = velx;    CvMat cvvely = vely;
	CvMat cvprev = prev;    CvMat cvcrnt = crnt;
	cvCalcOpticalFlowLK(&cvprev, &cvcrnt, winSize, &cvvelx, &cvvely);
	flow = convert2flow(velx, vely);
}

/*------------------------------------------------
Wrapper function that calls the  legacy implementation of the Horn-Shunk optical flow method (cvCalcOpticalFlowHS)
Input:
prev - Previous image, 8-bit, single-channel
crnt - crntent image, 8-bit, single-channel
usePrevoius - Flag that specifies whether to use the input velocity as initial approximations or not.
lambda – Smoothness weight. The smaller it is, the smoother optical flow map you get.
criteria – Criteria of termination of velocity computing
constructor: TermCriteria(int _type, int _maxCount, double _epsilon);
where _type can be: MAX_ITER, EPS or MAX_ITER+EPS
Output:
flow- matrix containing the optical flow vectors/pixel
Call example:
calcOpticalFlowHS( prev, crnt, 0, 0.1, TermCriteria(TermCriteria::MAX_ITER, 16, 0), flow);
-------------------------------------------------------------*/
void calcOpticalFlowHS(const Mat& prev, const Mat& crnt, int usePrevious, double lambda, TermCriteria criteria, Mat& flow)
{
	Mat velx(prev.size(), CV_32F), vely(prev.size(), CV_32F);
	CvMat cvvelx = velx;    CvMat cvvely = vely;
	CvMat cvprev = prev;    CvMat cvcrnt = crnt;
	cvCalcOpticalFlowHS(&cvprev, &cvcrnt, usePrevious, &cvvelx, &cvvely, lambda, criteria);
	flow = convert2flow(velx, vely);
}

/*-------------------------------------------------------------
Function used to display the optical flow vectors (LEGACY methods) filtered out by a length (modulus) threshold
Input:
name - destination (output) window name
gray -  background image to be displayed (usually the prev image)
flow - optical flow as a matrix of (x,y)
mult - scaling factor of the displayed image/window and of the optical flow vectors
minVel - threshold value (for vectors' modulus) for filtering out the displayed vectors
Call example:
showFlow (WIN_DST, prev, flow, 1, 4.0f, true, true, false);
-----------------------------------------------------------------*/
void showFlow(const string& name, const Mat& gray, const Mat& flow, int mult, float minVel,
	bool showImages, bool showVectors, bool showCircles)
{
	if (showImages)
	{
		Mat tmp, cflow;
		resize(gray, tmp, gray.size() * mult, 0, 0, INTER_NEAREST);
		cvtColor(tmp, cflow, CV_GRAY2BGR);

		// gain factor for the flow vectors display (usefull if the flow vectors are very small m2 > 1)
		const float m2 = 1.0f;

		for (int y = 0; y < flow.rows; ++y)
			for (int x = 0; x < flow.cols; ++x)
			{
				Point2f f = flow.at<Point2f>(y, x);

				if (f.x * f.x + f.y * f.y > minVel * minVel)
				{
					Point p1 = Point(x, y) * mult;
					Point p2 = Point(cvRound((x + f.x*m2) * mult), cvRound((y + f.y*m2) * mult));
					if (showVectors) // display flow vectors as green line segments
						line(cflow, p1, p2, CV_RGB(0, 255, 0));

					if (showCircles) // mark flow vectors' origins by a red circle
						circle(cflow, Point(x, y) * mult, 2, CV_RGB(255, 0, 0));
				}
			}

		imshow(name, cflow);
	}
}


/*-------------------------------------------------------------
Function used to display the SPARSE optical dense vectors filtered out by their status (1 or 0)
Input:
name - destination (output) window name
gray -  background image to be displayed (usually the prev image)
vector<Point2f> prev_pts
vector<Point2f> crnt_pts
vector<uchar> status;
vector<float> error;
mult - scaling factor of the displayed image/window and of the optical flow vectors
Call example:
showFlowSparse (WIN_DST, prev, prev_pts, crnt_pts, status, error, 2, true, true, true);
-----------------------------------------------------------------*/

void showFlowSparse(const string& name, const Mat& gray, const vector<Point2f>& prev_pts, const vector<Point2f>& crnt_pts,
	const vector<uchar>& status, const vector<float>& error, int mult,
	bool showImages, bool showVectors, bool showCircles)
{
	if (showImages)
	{
		Mat tmp, cflow;
		resize(gray, tmp, gray.size() * mult, 0, 0, INTER_NEAREST);
		cvtColor(tmp, cflow, CV_GRAY2BGR);

		for (int i = 0; i < prev_pts.size(); ++i)
		{
			if (showCircles) // mark flow vectors' origins by a red ircle
				circle(cflow, prev_pts[i] * mult, 4, CV_RGB(255, 0, 0));

			if (status[i]) //flow for crntent point i exists
			{
				Point2f p1 = prev_pts[i] * mult;
				//Point2f p2 = crnt_pts[i] * mult;
				Point2f p2 = Point(cvRound(crnt_pts[i].x * mult), cvRound(crnt_pts[i].y * mult));
				if (showVectors) // display flow vectors as green line segments
					line(cflow, p1, p2, CV_RGB(0, 255, 0));
				if (showCircles) // mark flow vectors' end by a blue circle
					circle(cflow, crnt_pts[i] * mult, 2, CV_RGB(0, 0, 255));
			}
		}

		imshow(name, cflow);
	}
}

/*-------------------------------------------------------------
Function used to display the DENSE optical flow vectors values using the Middlebury color encoding
Input:
name - destination (output) window name
gray -  background image to be displayed (usually the prev image)
flow - optical flow as a matrix of (x,y)
minVel - threshold value (for vectors' modulus) for filtering out the displayed vectors
Call example:
showFlowDense (WIN_DST, crnt, flow, minVel, true);
-----------------------------------------------------------------*/
void showFlowDense(const string& name, const Mat& gray, const Mat& flow, float minVel = 0.0, bool showImages = true)
{
	if (showImages)
	{
		Mat cflow;
		cvtColor(gray, cflow, CV_GRAY2BGR);
		unsigned char color[3];
		Mat_<Vec3b> _cflow = cflow;

		for (int y = 0; y < flow.rows; ++y)
			for (int x = 0; x < flow.cols; ++x)
			{
				Point2f f = flow.at<Point2f>(y, x);

				//COLOR_CONVENTION defined Functions.h
#if COLOR_CONVENTION == 0	//	0 - HSI convention
				computeColor(f.x, -f.y, color); // gnerates a color that encodes the orientation and modulus of the OF vector in (x,y)
#endif
#if COLOR_CONVENTION == 1	//	1 - Middlebury convetion
				computeColor(f.x, f.y, color); // gnerates a color that encodes the orientation and modulus of the OF vector in (x,y)
#endif
				float mod = sqrt(f.x * f.x + f.y * f.y); // modulus of the curent OF vector

				if (cflow.channels() == 3 && mod >= minVel)
				{
					_cflow(y, x)[0] = color[0];
					_cflow(y, x)[1] = color[1];
					_cflow(y, x)[2] = color[2];

				}
			}

		cflow = _cflow;

		imshow(name, cflow);
	}
}

/* Histogram display function - display a histogram using bars (simlilar tu L3 / PI)
Input:	
	name - destination (output) window name
	hist - pointer to the vector containing the histogram values
	hist_cols - no. of bins (elements) in the histogram = histogram image width
	hist_height - height of the histogram image
Call example: 
	showHistogram (WIN_HIST, hist_dir, 360, 200, true);
*/
void showHistogram (const string& name, int* hist, const int  hist_cols, const int hist_height, bool showImages = true)
{       
    if (showImages)
    {
		Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));

		//computes histogram maximum
		int max_hist = 0;
		for (int i=0; i<hist_cols; i++)
			if (hist[i] > max_hist)
				max_hist = hist[i];
		double scale = 1.0;  
		scale = (double)hist_height/max_hist;
		int baseline = hist_height -1;

   		for (int x=0; x < hist_cols; x++) {
			Point p1 = Point(x, baseline);
			Point p2 = Point(x, baseline - cvRound(hist[x]*scale));
			line(imgHist, p1, p2, CV_RGB(255, 0, 255));
		}
     
		imshow(name, imgHist);
    }
}

/* Opticol flow directions histogram display function - display a histogram using bars colored in Middlebury color coding 
Input:	
	name - destination (output) window name
	hist - pointer to the vector containing the histogram values
	hist_cols - no. of bins (elements) in the histogram = histogram image width
	hist_height - height of the histogram image
Call example: 
	showHistogramDir (WIN_HIST, hist_dir, 360, 200, true);
*/
void showHistogramDir (const string& name, int* hist, const int  hist_cols, const int hist_height, bool showImages = true)
{       
	unsigned char r, g, b;
	int R = 0;
	int G = 1;
	int B = 2;

    if (showImages)
    {
		Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));

		//computes histogram maximum
		int max_hist = 0;
		for (int i=0; i<hist_cols; i++)
			if (hist[i] > max_hist)
				max_hist = hist[i];
		double scale = 1.0;  
		scale = (double)hist_height/max_hist;
		int baseline = hist_height -1;

   		for (int x=0; x < hist_cols; x++) {
			Point p1 = Point(x, baseline);
			Point p2 = Point(x, baseline - cvRound(hist[x]*scale));
			r=HSI2RGB[x][R];
			g=HSI2RGB[x][G];
			b=HSI2RGB[x][B];
			line(imgHist, p1, p2, CV_RGB(r, g, b));
		}
     
		imshow(name, imgHist);
    }
}

// Calculeaza centru dreptunghi
Point RectCenter(Rect R)
{
	Point P;
	P.x = R.x + R.width / 2;
	P.y = R.y + R.height / 2;
	return P;
}

// Calculeaza aria dreptunghi
int RectArea(Rect R)
{
	return R.width*R.height;
}

// Deseneaza o cruce de dimensiune size x size peste punctil p
void DrawCross(Mat& img, Point p, int size, Scalar color, int thickness)
{
	line(img, Point(p.x - size / 2, p.y), Point(p.x + size / 2, p.y), color, thickness, 8);
	line(img, Point(p.x, p.y - size / 2), Point(p.x, p.y + size / 2), color, thickness, 8);
}

/* Etichetare / Detectie contur si calcul PGS folosind functii din OpenCV
Input:
	name - numele ferestrei in care s eva afisa rezultatul
	src - imagine binara rezultata in urma segmentarii (0/negru ~ fond; 255/alb ~ obiect)
	output_format = true : desenare obiecte pline ~ etichetare
					false : desenare conture ale obiectelor
Apel: 
	Labeling("Contur - functii din OpenCV", dst, false);
	*/
void Labeling(const string& name, const Mat& src, bool output_format)
{
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

				Scalar color(255,255,255);
				
				// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
				if (output_format) // desenare obiecte pline ~ etichetare
					drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
				else  //desenare contur obiecte
					drawContours(dst, contours, idx, color, 1, 8, hierarchy);

				Point center(xc, yc); //centrul de masa
				int radius = 5;

				// afisarea unor cercuri in jurul centrelor de masa
				//circle(final, center, radius,Scalar(255,255,355), 1, 8, 0);

				// afisarea unor cruci peste centrele de masa
				DrawCross(dst, center, 9, Scalar(255, 255, 355), 1);

				// https://en.wikipedia.org/wiki/Image_moment
				//calcul axa de alungire folosind momentele centarte de ordin 2
				double mc20p = m.m20 / m.m00 - xc*xc; // double mc20p = m.mu20 / m.m00;
				double mc02p = m.m02 / m.m00 - yc*yc; // double mc02p = m.mu02 / m.m00;
				double mc11p = m.m11 / m.m00 - xc*yc; // double mc11p = m.mu11 / m.m00;
				float teta = 0.5*atan2(2 * mc11p, mc20p - mc02p);
				float teta_deg = teta * 180 / PI;

				printf("ID=%d, arie=%.0f, xc=%0.f, yc=%0.f, teta=%.0f\n", idx, arie, xc, yc, teta_deg);

				int height = src.rows;
				int width = src.cols;
				Point p1, p2, p3, p4;
				float slope = tan(teta);
				float x1, x2, y1, y2;
				y1 = slope * (0 - xc) + yc;
				y2 = slope * (width - 1 - xc) + yc;
				x1 = (0 - yc) / slope + xc;
				x2 = (height - 1 - yc) / slope + xc;

				p1.x = 0;
				p1.y = y1;

				p2.x = width - 1;
				p2.y = y2;

				p3.x = x1;
				p3.y = 0;

				p4.x = x2;
				p4.y = height - 1;

				Point first, second;

				if (x1 >= 0 && x1 <= width - 1){
					first = p3;
				}

				if (x2 >= 0 && x2 <= width - 1){
					first = p4;
				}

				if (y1 >= 0 && y1 <= width - 1){
					second = p1;
				}
				if (y2 >= 0 && y2 <= width - 1){
					second = p2;
				}
				printf("p1.x=%d, p1.y=%.0f, p2.x=%0.f, p2.y=%0.f, p3.x=%.0f, p3.y=%.0f\n", p1.x, p1.y, p2.x, p2.y, p3.x,p3.y,p4.x,p4.y);
				line(dst, first, second, Scalar(255, 255, 255), 5, 8);
				
			}
		}
	}

	imshow(name, dst);
}
CascadeClassifier face_cascade; // cascade clasifier object for face
CascadeClassifier eyes_cascade; // cascade clasifier object for eyes
CascadeClassifier nose_cascade; // cascade clasifier object for face
CascadeClassifier mouth_cascade; // cascade clasifier object for eyes

void FaceDetectandDisplay(const string& window_name, Mat frame, int minFaceSize, int minEyeSize) {

	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(minFaceSize, minFaceSize) ); 
	for (int i = 0; i < faces.size(); i++){
		// get the center of the face
		Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
		// draw circle around the face
		ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0,  360, Scalar( 255, 0, 255 ), 4, 8, 0 );

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		//-- In each face (rectangular ROI), detect the eyes
		minFaceSize = 30;
		minEyeSize = minFaceSize / 5;

		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(minEyeSize, minEyeSize) ); 

		for (int j = 0; j < eyes.size(); j++){
			// get the center of the eye
			//atentie la modul in care se calculeaza pozitia absoluta a centrului ochiului
			// relativa la coltul stanga-sus al imaginii:
			Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5,  faces[i].y + eyes[j].y + eyes[j].height*0.5 );
			int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
			// draw circle around the eye
			circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 ); 
		}
	}
	imshow(window_name, frame); //-- Show what you got 
}



void FaceDetectandDisplayAll(const string& window_name, Mat frame, int minFaceSize, int minEyeSize, vector<Rect> facess){
	vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++)
	{
		// get the center of the face
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		// draw circle around the face
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0,
			360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);

		std::vector<Rect> eyes;
		//-- In each face (rectangular ROI), detect the eyes
		Rect eyes_rect;
		eyes_rect.x = faces[i].x;
		eyes_rect.y = faces[i].y + 0.2*faces[i].height;
		eyes_rect.width = faces[i].width;
		eyes_rect.height = 0.35*faces[i].height;
		Mat eyes_ROI = frame_gray(eyes_rect);
		minFaceSize = 20;
		minEyeSize = minFaceSize / 5;
		eyes_cascade.detectMultiScale(eyes_ROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(minEyeSize, minEyeSize));


		for (int j = 0; j < eyes.size(); j++)
		{
			eyes[j].x = eyes[j].x + faces[i].x;
			eyes[j].y = eyes[j].y + faces[i].y + 0.2*faces[i].height;


			// get the center of the eye
			//atentie la modul in care se calculeaza pozitia absoluta a centrului ochiului
			// relativa la coltul stanga-sus al imaginii:
			rectangle(frame, eyes[j], Scalar(255, 0, 0), 2, 8, 0);
			//Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5,
			//faces[i].y + eyes[j].y + eyes[j].height*0.5);
			//int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			// draw circle around the eye
			//circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}

		std::vector<Rect> nose;
		Rect nose_rect; //nose is the 40% ... 75% height of the face
		nose_rect.x = faces[i].x;
		nose_rect.y = faces[i].y + 0.4*faces[i].height;
		nose_rect.width = faces[i].width;
		nose_rect.height = 0.35*faces[i].height;
		Mat nose_ROI = frame_gray(nose_rect);
		
		nose_cascade.detectMultiScale(nose_ROI, nose, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(minEyeSize, minEyeSize));

		for (int j = 0; j < nose.size(); j++)
		{
			nose[j].x = nose[j].x + faces[i].x;
			nose[j].y = nose[j].y + faces[i].y + 0.4*faces[i].height;

			rectangle(frame, nose[j], Scalar(0, 255, 0), 2, 8, 0);
		}

		std::vector<Rect> mouth;
		Rect mouth_rect; //mouth is in the 70% ... 99% height of the face
		mouth_rect.x = faces[i].x;
		mouth_rect.y = faces[i].y + 0.7*faces[i].height;
		mouth_rect.width = faces[i].width;
		mouth_rect.height = 0.29*faces[i].height;
		Mat mouth_ROI = frame_gray(mouth_rect);
		minFaceSize = 20;
		minEyeSize = minFaceSize / 5;
		mouth_cascade.detectMultiScale(mouth_ROI, mouth, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(minEyeSize, minEyeSize));

		for (int j = 0; j < mouth.size(); j++)
		{
			mouth[j].x = mouth[j].x + faces[i].x;
			mouth[j].y = mouth[j].y + faces[i].y + 0.7*faces[i].height;

			rectangle(frame, mouth[j], Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	facess = faces;
	imshow(window_name, frame); //-- Show what you got
}


