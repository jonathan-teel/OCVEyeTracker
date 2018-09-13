/**
* [ Name ] OCVEyeTracker
* [ Desc ] Tracks coordinates of eye gaze through webcam using opencv
* [ Author ] Jonathan Teel 
**/

// linux command line
// g++ -ggdb `pkg-config --cflags opencv` -o `basename OCVEyeTracker.cpp .cpp` OCVEyeTracker.cpp `pkg-config --libs opencv`

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

void MatchEyeTemplate(Mat&, Mat&, Rect&);
void detectAndDisplay(Mat&, Mat&, Rect&);
void getColorInArea(Mat, Point, Point);
void makeGuess(int, int, int, int);
vector<String> sortQuads(int, int, int, int);

vector<int> screenInputs;
vector<float> weights;

class Perceptron
{
private:
    vector<int> expectedResults;
    int bias, result;
    float learningRate, output;

    float sigmoid(float f) {
        return 1 / (1 + exp(-f));
    }
    float sigmoidDerev(float f) {
        return sigmoid(f) * (1 - sigmoid(f));
    }
    float feedForward() {
        return sigmoid(screenInputs[0] * weights[0] + screenInputs[1] * weights[1]);
    }
    int checkOutput(float f) {
    	int res = 0;
    	for(int i=0; i<2; i++)
    	{
    		if(screenInputs[i] > expectedResults[i]+5)
        	{
        		res += -1;
        	}
			else if(screenInputs[i] < expectedResults[i]-5)
			{
				res += 1;
			}
			else 
				res += 0;
		}
    	result = res * f;    
    }
    void backProp()
    {
        for(int i=0; i<weights.size(); i++)
        {
            float ne = result * weights[i] * sigmoid(result);
            float wc = ne * result;
            weights[i] = weights[i] + wc * learningRate;
            if(weights[i] < 0 || weights[i] > 1) weights[i] = static_cast<float>(rand() / static_cast<float>(RAND_MAX));
        }
    }

public:
    Perceptron(float b, float lr): bias(b), learningRate(lr)
    {
        result = output = 0;
        srand(time(NULL));
        ifstream file;
        string sF;
        file.open("weights.txt");
        float inW[2];
        inW[0] = -1;
        float f;
        int c = 0;
        while(file >> f)
       	{
       		inW[c++] = f;
       	}
        if(inW[0] != -1)
        {
        	weights.push_back(inW[0]);
        	weights.push_back(inW[1]);
        }
        else
        {
        	weights.push_back(static_cast<float>(rand() / static_cast<float>(RAND_MAX)));
        	weights.push_back(static_cast<float>(rand() / static_cast<float>(RAND_MAX)));
        }
    }

    void nextExpectedResults(int x, int y)
    {
        expectedResults.clear();
        expectedResults.push_back(x);
        expectedResults.push_back(y);
    }

    void compute()
    {
		float f = feedForward();
		checkOutput(f);
		backProp();
    }

    void printWeights() {
        cout << weights[0] << ", " << weights[1] << endl;
    }

};

class Timer
{
private:
    unsigned long begTime;

public:
    Timer()
    {
        begTime = 0;
    }

    void start()
    {
        begTime = clock();
    }

    unsigned long elapsedTime()
    {
        return ((unsigned long) clock() - begTime) / CLOCKS_PER_SEC;
    }

    bool isTimeout(unsigned long seconds)
    {
        return seconds >= elapsedTime();
    }
};

class TrainingEnv
{

private:
    string wName;
    Point cPt;
    int thickness, startX, startY, screenWidth, screenHeight;
    unsigned int lineType, sWidth, sHeight;
    unsigned long cTimeLimit;
    Timer t;

public:
    TrainingEnv(int sw, int sh): screenWidth(sw), screenHeight(sh)
    {
        wName = "Training Env";
        namedWindow(wName, CV_WINDOW_NORMAL);
        setWindowProperty(wName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        resizeWindow(wName, sw, sh);
        thickness = -1;
        lineType = 8;
        cTimeLimit = 2;
        sWidth = sHeight = 0;
        startX = startY = 20;
        cPt.x = startX;
        cPt.y = startY;
        srand(time(NULL));
    }

    int run(Perceptron &p, VideoCapture &capture)
    { 
		Mat frame, eyeTemplate;
		Rect eyeRect;
        if(!capture.isOpened()) return -1;
        sHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
        sWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
        p.nextExpectedResults(cPt.x, cPt.y);
        t.start();
        while(true)
        {  
            capture >> frame;
	    
			if(!frame.empty())
			{
				Mat frameGray;
				cvtColor(frame, frameGray, CV_BGR2GRAY);

				if(eyeRect.width == 0 && eyeRect.height == 0)
				{
					detectAndDisplay(frameGray, eyeTemplate, eyeRect);
				}
				else
				{
					imwrite("eyes.jpg", eyeTemplate);
					MatchEyeTemplate(frameGray, eyeTemplate, eyeRect);
					rectangle(frame, eyeRect, CV_RGB(0,255,0));
				}
            }
            else
            {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            p.compute();
            if(t.elapsedTime() >= cTimeLimit) getNewCPt();
			circle(frame, cPt, 10, Scalar(0, 0, 250), thickness, lineType);
			imshow(wName, frame);
			if(waitKey(1) == 27) 
			{
				destroyWindow(wName);
				break;
			}
        }

    }

    void getNewCPt()
    {
        if(startX > 500)
        {
            startX = 20;
            startY += 60;
        }
        else
            startX += 60;
        cPt.x = startX;
        cPt.y = startY;
        t.start();
    }
};

String faceCascadeName = "lbpcascade_frontalface.xml";
String eyesCascadeName = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;
const char* windowName = "Eye Tracker";
const char* pupilWindow = "Pupil";
const char* eyeWindow = "Eye Area";
const char* origPupilWindow = "Original Pupil Area";
Mat eyesROI;
bool assigned = false;
RNG rng(12345);
int tValue = 35;
vector<int> qOne, qTwo, qThree, qFour;

// currently a constant of screen resolution, this is an OS function to get dynamically, could be passed as param
const int width = 1366;
const int height = 768;

// Compares the captured eye to the source image
void MatchEyeTemplate(Mat& sourceImg, Mat& templateImg, Rect& rect)
{
   Size size(rect.width*2, rect.height*2);
   Rect window(rect+size-Point(size.width/2, size.height/2));
   
   window &= Rect(0, 0, sourceImg.cols, sourceImg.rows);
   
   Mat dst(window.width - templateImg.rows + 1, window.height - templateImg.cols + 1, CV_32FC1);
   matchTemplate(sourceImg(window), templateImg, dst, CV_TM_SQDIFF_NORMED);

   double minval, maxval;
   Point minloc, maxloc;
   minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);

   if (minval <= 0.2)
   {
       rect.x = window.x + minloc.x;
       rect.y = window.y + minloc.y;
       
       getColorInArea(sourceImg(window), minloc, Point(minloc.x+templateImg.cols, minloc.y+templateImg.rows) );
   }
   else
       rect.x = rect.y = rect.width = rect.height = 0;
}

// detect face , eyes , and track
void detectAndDisplay(Mat& frame, Mat& templ, Rect& rect)
{
    vector<Rect> faces, eyes;
    
    faceCascade.detectMultiScale(frame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30,30));
    
    for(int i = 0; i < faces.size(); i++)
    {
		Mat face = frame(faces[i]);
		eyesCascade.detectMultiScale(face, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(20,20));
	
		if(eyes.size())
		{
			rect = eyes[0] + Point(faces[i].x, faces[i].y);
			templ = frame(rect);
		}
    }
}

// get pixel color in area
void getColorInArea(Mat image, Point p1, Point p2)
{
    Mat roiImg, pup, tPup, p;
    
    roiImg = image(Rect(p1,p2));
    
    cvtColor(roiImg, roiImg, CV_GRAY2BGR);

    // expected pupil area, center of image +-20
    int offset = 20;
    Point px((roiImg.rows/2)-offset+5, (roiImg.cols/2)-offset);
    Point py((roiImg.rows/2)+offset+5, (roiImg.cols/2)+offset);

    // get pupil from eye area , * extra widows DEBUG only
    pup = roiImg(Rect(px, py));
    tPup = pup;

    cvtColor(tPup, tPup, CV_BGR2GRAY);
    cvtColor(pup, pup, CV_BGR2GRAY);

    equalizeHist(pup, pup);
    threshold(pup, pup, tValue, 255, THRESH_BINARY);
    int mRow = pup.rows; // mRow = 20
    int mCol = pup.cols; // mCol = 20
    // quadrant total black pixel count
    int q1=0, q2=0, q3=0, q4=0;

    qOne.clear();
    qTwo.clear();
    qThree.clear();
    qFour.clear();

    for(int i=0; i<4; i++)
    {
        qOne.push_back(0);
        qTwo.push_back(0);
        qThree.push_back(0);
        qFour.push_back(0);
    }

    // loop thru image, checking pixel colors in each quadrant , if not white ++
    for(register int i=0; i<mRow; i++)
    {
        for(register int j=0; j<mCol; j++)
        {
            int c = static_cast<int>(pup.at<unsigned char>(i, j));
            if(i<=mRow/2 && j<=mCol/2 && c!=255)
            {
                q2++;
            }
            else if(i>mRow/2 && j<=mCol/2 && c!=255) q1++;
            else if(i<=mRow/2 && j>mCol/2 && c!=255) q3++;
            else if(i>mRow/2 && j>mCol/2 && c!=255) q4++;
        }
    }
    makeGuess(q1, q2, q3, q4);
}

// return string format of quad's sorted black pixel count , ascending
vector<String> sortQuads(int q1, int q2, int q3, int q4)
{
    vector<String> res;
    // initialize result vector to " "
    for(int i=0; i<4; i++) res.push_back(" ");
    vector<int> q;
    q.push_back(q1);
    q.push_back(q2);
    q.push_back(q3);
    q.push_back(q4);
    sort(q.begin(), q.end());
    // convert sorted ints to sorted strings
    for(register int i=0; i<q.size(); i++)
    {
        String nQ;
        if(q[i] == q1) nQ = "q1";
        else if(q[i] == q2) nQ = "q2";
        else if(q[i] == q3) nQ = "q3";
        else if(q[i] == q4) nQ = "q4";
        res[i] = nQ;
    }
    return res;
}

// return second largest quadrant
int getSecondLargest(int q1, int q2, int q3, int q4)
{
    int arr[4] = {q1, q2, q3, q4};
    sort(arr, arr+4);
    return arr[1];
}

// make guess at eye location 
void makeGuess(int q1, int q2, int q3, int q4)
{
    String res = "";
    vector<string> sorted = sortQuads(q1, q2, q3, q4);

    if(sorted[3] == "q2") 
    {
    	res = "top right";
    	screenInputs[0] = width * weights[0] + 10;
    	screenInputs[1] = height/4 * weights[1] - 10;
    }
    else if(sorted[3] == "q1") 
    {
    	res = "top left";
    	screenInputs[0] = width/4 * weights[0] + 10;
    	screenInputs[1] = height/4 * weights[1] + 10;
    }
    else if(sorted[3] == "q3")
    {
        res = "bottom left";
        screenInputs[0] = width * weights[0] + 10;
        screenInputs[1] = height/5 * weights[1] + 10;

    }
    else if(sorted[3] == "q4") 
    {
    	res = "bottom right";
    	screenInputs[0] = width * weights[0] - 10;
    	screenInputs[1] = height * weights[1] - 10;
    }
    
    cout << res;
}


// main
int main( int argc, const char* argv[] )
{
    Mat trackEye;
    Rect eyeRect;
    Mat frame, eyeTemplate;
    for(int i=0; i<2; i++) screenInputs.push_back(0);
    Perceptron p(-1.0f, 0.5f); 
    //TrainingEnv t(1366, 760);
    long maxPrintTime = 0.5;
    Timer tt;
    
    VideoCapture cap(0);

    //namedWindow(eyeWindow, 0);
    //resizeWindow(eyeWindow, 500, 500);

    //namedWindow(origPupilWindow, 0);
    //resizeWindow(origPupilWindow, 350, 350);

    if(!faceCascade.load( faceCascadeName ))
    {
        printf("--(!)Error loading face cascade\n");
        return -1;
    }
    if(!eyesCascade.load( eyesCascadeName ))
    {
        printf("--(!)Error loading eyes cascade\n");
        return -1;
    }

    // * run a couple times to train at beginning
    //t.run(p, cap);
    
    // save the weights
    ofstream file;
    file.open("weights.txt");
    for(int i=0; i<weights.size(); i++) file << weights[i] << " ";
     
    if(!cap.isOpened()) return -1;
    namedWindow(pupilWindow, 0);
    setWindowProperty(pupilWindow, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    resizeWindow(pupilWindow, width, height);
    cout << "Entering normal environment.. " << endl;
    tt.start();
    while(true) {
        cap >> frame;
        if(!frame.empty()) 
		{		
			Mat frameGray;
			cvtColor(frame, frameGray, CV_BGR2GRAY);
			if(eyeRect.width == 0 && eyeRect.height == 0)
			{
				detectAndDisplay(frameGray, eyeTemplate, eyeRect);
			}
			else
			{
				imwrite("eyes.jpg", eyeTemplate);
				MatchEyeTemplate(frameGray, eyeTemplate, eyeRect);
				rectangle(frame, eyeRect, CV_RGB(0,255,0));
			}
			if(tt.elapsedTime() >= maxPrintTime)
			{
				//cout << screenInputs[0] << "   :   " << screenInputs[1] << endl;
				tt.start();
			}
			circle(frame, Point(screenInputs[0], screenInputs[1]), 10, Scalar(0, 0, 250), -1, 8);
			imshow(pupilWindow, frame);
        }
        else 
		{
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        if(waitKey(100) == 112) 
		{
            //destroyWindows();
            cout << "Exiting Application.." << endl;
            break;
        }
    }
    cap.release();    

    return 0;
}
