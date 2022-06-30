#include <iostream>
#include <opencv2/opencv.hpp>

#define DELAY 1
#define VIDEO_WIDTH 1920
#define VIDEO_HEIGHT 1080

int main(int, char **)
{
	cv::Mat frame;
	int cameraInd = 0;
	//	cv::VideoCapture video = cv::VideoCapture(cameraInd); // initialize video capture
	//	const int deviceID = 0; // 0 = open default camera
	//	const int apiID = cv::CAP_ANY;    // 0 = auto detect default API
	int capNum = 0; // capture counting number
	//
	//	video.open(deviceID, apiID);    // open selected camera using selected API
	auto video = cv::VideoCapture(cameraInd);
	if (!video.isOpened())
	{
		std::cerr << "Error: cannot open device!/n";
		return -1;
	}

	// set the width and height of the frame, otherwise the default will be 640x480
	video.set(CV_CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH);
	video.set(CV_CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT);

	// set framerate of the videoCapture
	//	video.set(CV_CAP_PROP_FPS, 30);

	int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	video.set(CV_CAP_PROP_FOURCC, fourcc);

	std::cout << "Start grabbing: \n Press 's' to save image and press 'q' to terminate.\n";

	cv::namedWindow("Live", cv::WINDOW_AUTOSIZE);
	while (true)
	{
		// wait for a new frame from camera and store it into frame
		video.read(frame); // or videoCapture >> frame;
		if (frame.empty()) { continue; }

		cv::imshow("Live", frame);
		if (cv::waitKey(DELAY) == 's' || cv::waitKey(DELAY) == 'S')
		{
			std::string imName = "../raw_images/" + std::to_string(++capNum) + ".png";
			cv::imwrite(imName, frame);
			std::cout << "Wrote to : " << imName << std::endl;
		}
		if (cv::waitKey(DELAY) == 'q' || cv::waitKey(DELAY) == 'Q') { break; }
	}
	video.release();
	// the camera will be destroyed automatically in VideoCapture destructor.
	return 0;
}