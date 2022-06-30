#include <iostream>
#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>

#include "ImageWrapper.h"

int centerX = 0;
int centerY = 0;
int dis = 0;
int scaleRatio = 0;
double topCutoff = 0;
double bottomCutoff = 0;
int recordedWidth, recordedHeight;
int calibratedWidth, calibratedHeight;
double recordRatioX, recordRatioY;
// parameters for choosing ROI
//unsigned int topMargin = 200, bottomMargin = 200;
//unsigned int leftMargin = 300, rightMargin = 300;

/**
 * @brief 显示unwrapped的图像并储存
 * @details 按s储存图片，按q退出程序
 * @param camNum 相机设备的id，从0开始
 * @param imgSaveNum 起始的图片编号，默认为0
 * @param scale 尺度因子，默认为1
 */
void streamImage(int camNum, unsigned int imgSaveNum = 0, unsigned int scale = 1);

int main(int argc, char *argv[])
{
	if (3 != argc)
	{
		std::cerr << "Error: not enough arguments.\nUsage: [" << argv[0]
				  << "] [initial image number to save] [camera number]\n";
		return 1;
	}

	Json::Value rootValue;
	std::ifstream ifsRoot("../config/lawn_mover.json", std::ios::binary);
	if (!ifsRoot.is_open())
	{
		std::cerr << "Error: cannot open root file!\n";
		return -1;
	}
	ifsRoot >> rootValue;
	ifsRoot.close();

	std::string cameraConfig = rootValue["current_camera"].asString();
	recordedWidth = rootValue["camera_record_res"][0].asInt();
	recordedHeight = rootValue["camera_record_res"][1].asInt();
	scaleRatio = rootValue["display_ratio"].asInt();

	Json::Value cameraValue;
	std::ifstream ifsCamera("../config/" + cameraConfig + ".json");
	if (!ifsCamera.is_open())
	{
		std::cerr << "Error: cannot open camera file!\n";
		return -1;
	}
	ifsCamera >> cameraValue;
	ifsCamera.close();

	calibratedWidth = cameraValue[1]["raw_resolution"][0].asInt();
	calibratedHeight = cameraValue[1]["raw_resolution"][1].asInt();
	recordRatioX = (double) calibratedWidth / recordedWidth;
	recordRatioY = (double) calibratedHeight / recordedHeight;

	if (recordRatioX != recordRatioY)
	{
		std::cerr << "Error: Recorde image aspect ratio different from calibrated file!" << std::endl;
		return -1;
	}

	dis = (int) ((double) cameraValue[1]["distance"].asInt() * 2 / scaleRatio / recordRatioX);
	centerX = (int) ((double) cameraValue[1]["center_x"].asInt() / scaleRatio / recordRatioX);
	centerY = (int) ((double) cameraValue[1]["center_y"].asInt() / scaleRatio / recordRatioY);

	topCutoff = cameraValue[1]["top_cutoff"].asDouble();
	bottomCutoff = cameraValue[1]["bottom_cutoff"].asDouble();

	unsigned int imgSaveNum = std::stoi(argv[1]);
	int camNum = std::stoi(argv[2]);
	streamImage(camNum, imgSaveNum);

	return 0;
}

void streamImage(int camNum, unsigned int imgSaveNum, unsigned int scale)
{
	auto video = cv::VideoCapture(camNum);
	if (!video.isOpened())
	{
		std::cerr << &"Error: cannot open camera: "[camNum] << std::endl;
		return;
	}

	// set openCV capture resolution
	video.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	video.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);

	int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	video.set(CV_CAP_PROP_FOURCC, fourcc);

	cv::Mat frame, resizedImage;

	ImageWrapper wrapper;
	bool res = wrapper.initImageWrapper(dis, dis);
	if (!res)
	{
		std::cerr << "Error: Failed to init image wrapper!" << std::endl;
		return;
	}
	std::cout << "Image wrapper init success!\n";

	bool isNotEnd = true;

	while (isNotEnd)
	{
		auto tStart = std::chrono::steady_clock::now();
		isNotEnd = video.read(frame);
		auto tEnd = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart);
		std::cout << "Time spent: " << duration.count() << std::endl;
		if (frame.empty()) { continue; }
		if (1 != scale) { cv::resize(frame, frame, cv::Size(), scale, scale, cv::INTER_LINEAR); }

		cv::resize(frame, resizedImage, cv::Size((int) (frame.cols / scaleRatio), (int) (frame.rows / scaleRatio)));

		cv::Rect myROI(centerX - dis / 2, centerY - dis / 2, dis, dis);
		cv::Mat croppedImage = resizedImage(myROI);

		cv::Mat image = wrapper.getImage(croppedImage);

		cv::Rect myROI2(0, (int) (image.rows / topCutoff), image.cols, (int) (image.rows / bottomCutoff));
		cv::Mat finalImage = image(myROI2);


		cv::imshow("final", finalImage);

		// cut margins out
		// cv::Rect finalROI(leftMargin, topMargin, finalImage.rows - leftMargin- rightMargin, finalImage.cols-topMargin-bottomMargin);
		// finalImage = finalImage(finalROI);

		// save unwrapped images
		if (cv::waitKey(1) == 's' || cv::waitKey(1) == 'S')
		{
			std::string imName = "../raw_images/" + std::to_string(++imgSaveNum) + ".png";
			cv::imwrite(imName, finalImage);
			std::cout << "Wrote to: " << imName << std::endl;
		}
		else if (cv::waitKey(1) == 'q' || cv::waitKey(1) == 'Q')
		{
			std::cout << "Quitting program." << std::endl;
			return;
		}
		cv::waitKey(1);

	}
}
