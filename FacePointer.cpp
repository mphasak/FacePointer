#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/photo/photo.hpp>

int noseMinX = std::numeric_limits<int>::max();
int noseMinY = std::numeric_limits<int>::max();
int noseMaxX = std::numeric_limits<int>::min();
int noseMaxY = std::numeric_limits<int>::min();

cv::Point lastMousePos;

const int numFramesToCalibrateNoseRangeWith = 200;
int calibrationFrameNum = 0;

/**
* Given a rectangle denoting a face, and one denoting a nose, return a mouse position
*/
void faceNoseToMousePos(const cv::Size frameSize, const cv::Rect &faceRect, const cv::Rect &noseRect, cv::Point &mousePos)
{
	// Use the first few frames for range calibration
	if (calibrationFrameNum < numFramesToCalibrateNoseRangeWith) {
		if (noseRect.x > noseMaxX) noseMaxX = noseRect.x;
		if (noseRect.y > noseMaxY) noseMaxY = noseRect.y;
		if (noseRect.x < noseMinX) noseMinX = noseRect.x;
		if (noseRect.y < noseMinY) noseMinY = noseRect.y;
		calibrationFrameNum++;
		//std::cout << "calibrating: " << noseMinX << "," << noseMinY << "\t" << noseMaxX << "," << noseMaxY << std::endl;
	}
	else {
		mousePos.x = frameSize.width - int(frameSize.width  * (1.0 * (noseRect.x - noseMinX) / (noseMaxX - noseMinX)));
		mousePos.y = int(frameSize.height * (1.0 * (noseRect.y - noseMinY) / (noseMaxY - noseMinY)));
		//std::cout << "setting: " << noseRect.x << "," << noseRect.y << "\t" << mousePos.x << "," << mousePos.y << std::endl;
		lastMousePos = mousePos;
	}
}

/**
* Given a rectangle denoting a face, and a point denoting a chosen feature point (usually nostril), return a mouse position
*/
void featurePointToMousePos(const cv::Size frameSize, const cv::Rect &faceRect, const cv::Point &featurePos, cv::Point &mousePos)
{
	// Use the first few frames for range calibration
	if (calibrationFrameNum < numFramesToCalibrateNoseRangeWith) {
		if (featurePos.x > noseMaxX) noseMaxX = featurePos.x;
		if (featurePos.y > noseMaxY) noseMaxY = featurePos.y;
		if (featurePos.x < noseMinX) noseMinX = featurePos.x;
		if (featurePos.y < noseMinY) noseMinY = featurePos.y;
		calibrationFrameNum++;
		//std::cout << "calibrating: " << noseMinX << "," << noseMinY << "\t" << noseMaxX << "," << noseMaxY << std::endl;
	}
	else {
		mousePos.x = frameSize.width - int(frameSize.width  * (1.0 * (featurePos.x - noseMinX) / (noseMaxX - noseMinX)));
		mousePos.y = int(frameSize.height * (1.0 * (featurePos.y - noseMinY) / (noseMaxY - noseMinY)));
		//std::cout << "setting: " << noseRect.x << "," << noseRect.y << "\t" << mousePos.x << "," << mousePos.y << std::endl;
		lastMousePos = mousePos;
	}
}

int main()
{

	cv::VideoCapture cap;
	cap.open(0);

	cap.set(CV_CAP_PROP_FPS, 30);

	if (!cap.isOpened())
	{
		std::cerr << "***Could not initialize capturing...***\n";
		std::cerr << "Current parameter's value: \n";
		return -1;
	}

	cv::CascadeClassifier faceDetector;

	//std::string faceDetectorFilename = "haarcascade_frontalface_default.xml";
	std::string faceDetectorFilename = "lbpcascade_frontalface.xml";

	try {
		faceDetector.load(faceDetectorFilename);
	}
	catch (cv::Exception e) {
		std::cerr << "Exception loading detector file (";
		std::cerr << faceDetectorFilename;
		std::cerr << ")" << std::endl;
	}
	if (faceDetector.empty()) {
		std::cerr << "ERROR: face detector file was empty" << std::endl;
		exit(1);
	}

	// Face classifier params
	const float searchScaleFactor_face = 1.2f; // How much the image size is reduced at each image scale in the viola-jones search
	int minNeighbors_face = 5; // Reliability threshold: how many nbr rects each candidate haar rect needs -- typically 3
	const int flags_face = cv::CASCADE_FIND_BIGGEST_OBJECT | cv::CASCADE_DO_ROUGH_SEARCH;  // only return the one biggest nose
	const cv::Size minFeatureSize_face(50, 50); // Smallest object size


	cv::CascadeClassifier noseDetector;

	std::string noseDetectorFilename = "haarcascade_mcs_nose.xml";

	try {
		noseDetector.load(noseDetectorFilename);
	}
	catch (cv::Exception e) {
		std::cerr << "Exception loading detector file (";
		std::cerr << noseDetectorFilename;
		std::cerr << ")" << std::endl;
	}
	if (noseDetector.empty()) {
		std::cerr << "ERROR: nose detector file was empty" << std::endl;
		exit(1);
	}


	// Initialize the adaptive contrast equalizer
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	int claheClipLimit = 2;
	cv::Size claheTileGridSize(8, 8);
	clahe->setClipLimit(claheClipLimit);
	clahe->setTilesGridSize(claheTileGridSize);

	// Classifier parameters    
	const float searchScaleFactor_nose = 1.2f; // How much the image size is reduced at each image scale in the viola-jones search
	int minNeighbors_nose = 3; // Reliability threshold: how many nbr rects each candidate haar rect needs -- typically 3
	const int flags_nose = cv::CASCADE_FIND_BIGGEST_OBJECT | cv::CASCADE_DO_ROUGH_SEARCH;  // only return the one biggest nose
	const cv::Size minFeatureSize_nose(15, 15); // Smallest object size


	// LK optical flow
	bool nightMode = false;
	bool removePoint = false;
	cv::Point2f pt;
	cv::TermCriteria termcrit = cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	cv::Size winSize = cv::Size(10, 10);
	int maxOpticalFlowPoints = 5;
	cv::Mat previousFlowFrame;
	cv::Mat flowMask;
	std::vector<cv::Point2f> points[2];
	//std::vector<uchar> status;
	//std::vector<float> err;
	int framesBetweenLKReInit = 30;
	int maxIntensityToTrack = 50;  // to avoid things like glare moving the pointer, only track dark features
	cv::Point2f centerOfMass;
	bool haveStableChosenPoint = false; // this will be true once a feature is chosen to track, and false when it's lost
	size_t chosenPointIdx = 0; // to store the index of the chosen feature, for tracking if it's lost or not

	cv::Mat frame, grayscaleFrame;
	int frameIdx = 0;
	int flowIdx = 0;
	size_t exposureStep = 0; // for cycling between different exposures
	bool doEqualizeHist = false;

	int glareThresholdFactor = 9;
	double glareInpaintRadius = 3.0;



	while (1){

		cap >> frame;
		if (frame.empty()){
			std::cerr << "frame is empty" << std::endl;
			break;
		}

		if (frame.channels() == 3) {
			cvtColor(frame, grayscaleFrame, CV_BGR2GRAY);
		}
		else if (frame.channels() == 4) {
			cvtColor(frame, grayscaleFrame, CV_BGRA2GRAY);
		}
		else {
			grayscaleFrame = frame;
		}

		const int DETECTION_WIDTH = 320; // shrink the image before running detection for performance
		cv::Mat smallImg, equalizedSmallImg;
		float scale = grayscaleFrame.cols / (float)DETECTION_WIDTH;
		if (grayscaleFrame.cols > DETECTION_WIDTH) {
			// Shrink the image while keeping the aspect ration
			int scaledHeight = cvRound(grayscaleFrame.rows / scale);
			resize(grayscaleFrame, smallImg, cv::Size(DETECTION_WIDTH, scaledHeight));
		}
		else {
			smallImg = grayscaleFrame;
		}

		/*
		// Filter glare -- commenting this out since I wasn't getting good results, but maybe it can be better tuned
		int greyscaleThreshold = int(255 * 0.1 * glareThresholdFactor);
		cv::Mat glareMask = smallImg >= greyscaleThreshold;
		glareMask *= 0.5;
		smallImg -= glareMask;
		//cv::inpaint(smallImg, glareMask, smallImg, glareInpaintRadius, CV_INPAINT_TELEA);
		*/

		// Equalize histogram
		//equalizeHist(smallImg, smallImg);

		// Adaptive contrast equalizer
		clahe->setTilesGridSize(claheTileGridSize);
		clahe->setClipLimit(claheClipLimit);
		clahe->apply(smallImg, smallImg);


		//cv::Mat g1, g2;
		//GaussianBlur(smallImg, g1, cv::Size(1,1), 0);
		//GaussianBlur(smallImg, g2, cv::Size(3,3), 0);
		//frame = g1 - g2;

		// Detect 
		std::vector<cv::Rect> faces;
		faceDetector.detectMultiScale(smallImg,
			faces,
			searchScaleFactor_face,
			minNeighbors_face,
			flags_face, // flags
			minFeatureSize_face);
		if (doEqualizeHist) {
			equalizeHist(smallImg, smallImg);
		}

		if (faces.size() > 0) {

			// Trim the face rect so we only look near the center of the face
			float scaleFactor = 0.6f;
			int trimmedFaceWidth = int(faces[0].width * scaleFactor);
			int trimmedFaceHeight = int(faces[0].height * scaleFactor);
			int trimmedFaceX = int(faces[0].x + ((faces[0].width - trimmedFaceWidth) / 2.0));
			int trimmedFaceY = int(faces[0].y + ((faces[0].height - trimmedFaceHeight) / 2.0));
			cv::Rect trimmedFaceRect(trimmedFaceX, trimmedFaceY, trimmedFaceWidth, trimmedFaceHeight);


			if (!haveStableChosenPoint) {
				// Mask out an ROI based on the face rect -- run LK only in there
				flowMask = cv::Mat::zeros(smallImg.size(), CV_8UC1);
				flowMask(trimmedFaceRect) = 255;

				// Update the tracking points
				cv::goodFeaturesToTrack(smallImg,
					points[1],				// the output points
					maxOpticalFlowPoints,
					0.01,					// minimal accepted quality of image corners
					10,						// min. possible Euclidean distance between pts
					flowMask,				// optional ROI mask
					3,						// (average) block size
					0,						// use Harris detector
					0.04                    // free parameter of the Harris detector if applicable
					);
				cv::cornerSubPix(smallImg(trimmedFaceRect), points[1], winSize, cv::Size(-1, -1), termcrit);
				points[0].clear();
			}

			flowIdx++;
			if (flowIdx == framesBetweenLKReInit) flowIdx = 0;

			// Enlarge the results if the image was temporarily shrunk. 
			if (frame.cols > smallImg.cols) {
				for (int i = 0; i < (int)faces.size(); i++) {
					faces[i].x = cvRound(faces[i].x * scale);
					faces[i].y = cvRound(faces[i].y * scale);
					faces[i].width = cvRound(faces[i].width * scale);
					faces[i].height = cvRound(faces[i].height * scale);
				}
			}
			if (frame.cols > smallImg.cols) {
				trimmedFaceRect.x = cvRound(trimmedFaceRect.x * scale);
				trimmedFaceRect.y = cvRound(trimmedFaceRect.y * scale);
				trimmedFaceRect.width = cvRound(trimmedFaceRect.width * scale);
				trimmedFaceRect.height = cvRound(trimmedFaceRect.height * scale);
			}
			// If the object is on a border, keep it in the image. 
			for (int i = 0; i < (int)faces.size(); i++) {
				if (faces[i].x < 0)
					faces[i].x = 0;
				if (faces[i].y < 0)
					faces[i].y = 0;
				if (faces[i].x + faces[i].width > frame.cols)
					faces[i].x = frame.cols - faces[i].width;
				if (faces[i].y + faces[i].height > frame.rows)
					faces[i].y = frame.rows - faces[i].height;
			}

			//rectangle(frame, trimmedFaceRect, cv::Scalar(0, 255, 0)); // outline the face
			//rectangle(frame, faces[0], cv::Scalar(200, 0, 0));
		}

		if (previousFlowFrame.empty())
			smallImg.copyTo(previousFlowFrame);

		if (!points[0].empty()) {  // LK flow needs initialization

			std::vector<uchar> status;
			std::vector<float> err;

			calcOpticalFlowPyrLK(previousFlowFrame, smallImg, points[0], points[1], status, err, winSize);

			cv::Point2f chosenPointerPoint;
			if ((status.size() > chosenPointIdx) && status[chosenPointIdx])
			{
				// The previously-chosen point is still being tracked, so choose it again
				chosenPointerPoint = points[1][chosenPointIdx];
			}
			else
			{
				// Choose a new point to track 

				size_t i, k;
				// Compute center of mass
				float totalX = 0;
				float totalY = 0;
				for (i = 0; i < points[1].size(); i++) { totalX += points[1][i].x; totalY += points[1][i].y; }
				centerOfMass = cv::Point2f(totalX / points[1].size(), totalY / points[1].size());

				cv::circle(frame, centerOfMass * scale, 3, cv::Scalar(255, 0, 0), -1, 8);

				//double maxDisplacement = 0;
				double maxWeight = 0;
				for (i = k = 0; i < points[1].size(); i++) {
					if (!status[i])
						continue;

					points[1][k++] = points[1][i];


					// Compute a fitness-for-pointing weight for each LK feature point, keeping
					// the one with the highest weight
					int curValue = int(smallImg.at<uchar>(points[1][i]));
					double distanceFromCenter = cv::norm(centerOfMass - points[1][i]);
					double curWeight = ((255 - curValue)*(255 - curValue)) / distanceFromCenter;
					if (curWeight > maxWeight) {
						maxWeight = curWeight;
						chosenPointerPoint = points[1][i];
						chosenPointIdx = i;
					}
					//if (int(smallImg.at<uchar>(points[1][i])) < maxIntensityToTrack) {
					//double displacement = cv::norm(points[0][i] - points[1][i]);
					//if (displacement > maxDisplacement) {
					//	maxDisplacement = displacement;
					//	chosenPointerPoint = points[1][i];
					//}

					cv::circle(frame, points[1][i] * scale, 3, cv::Scalar(0, 255, 0), -1, 8);
					//}

				}
				points[1].resize(k);
			}
			cv::circle(frame, chosenPointerPoint * scale, 3, cv::Scalar(0, 0, 255), -1, 8);
			cv::Point mousePos(0, 0);
			featurePointToMousePos(cv::Size(frame.cols, frame.rows), faces[0], chosenPointerPoint * scale, mousePos);
		}

		std::swap(points[1], points[0]);
		cv::swap(previousFlowFrame, smallImg);
		/*
   //	  // Nose tracking using the opencv nose detector
   //     std::vector<cv::Rect> noses;
   //     if (faces.size() > 0) {

   //         cv::Size maxFeatureSize_nose(faces[0].width / 2, faces[0].height / 2); // constrain nose size based on face size
   //         noseDetector.detectMultiScale(smallImg(faces[0]),
   //                                       noses,
   //                                       searchScaleFactor_nose,
   //                                       minNeighbors_nose,
   //                                       flags_nose, // flags
   //                                       minFeatureSize_nose,
   //                                       maxFeatureSize_nose);

   //         // Enlarge the results if the image was temporarily shrunk.
   //         if (frame.cols > smallImg.cols) {
   //             for (int i = 0; i < (int)faces.size(); i++ ) {
   //                 faces[i].x = cvRound(faces[i].x * scale);
   //                 faces[i].y = cvRound(faces[i].y * scale);
   //                 faces[i].width = cvRound(faces[i].width * scale);
   //                 faces[i].height = cvRound(faces[i].height * scale);
   //             }
   //             for (int i = 0; i < (int)noses.size(); i++ ) {
   //                 noses[i].x = faces[i].x + cvRound(noses[i].x * scale);
   //                 noses[i].y = faces[i].y + cvRound(noses[i].y * scale);
   //                 noses[i].width = cvRound(noses[i].width * scale);
   //                 noses[i].height = cvRound(noses[i].height * scale);
   //             }
   //         }
   //         // If the object is on a border, keep it in the image.
   //         for (int i = 0; i < (int)faces.size(); i++ ) {
   //             if (faces[i].x < 0)
   //                 faces[i].x = 0;
   //             if (faces[i].y < 0)
   //                 faces[i].y = 0;
   //             if (faces[i].x + faces[i].width > frame.cols)
   //                 faces[i].x = frame.cols - faces[i].width;
   //             if (faces[i].y + faces[i].height > frame.rows)
   //                 faces[i].y = frame.rows - faces[i].height;
   //         }

   //         for (int i = 0; i < (int)noses.size(); i++ ) {
   //             if (noses[i].x < 0)
   //                 noses[i].x = 0;
   //             if (noses[i].y < 0)
   //                 noses[i].y = 0;
   //             if (noses[i].x + noses[i].width > frame.cols)
   //                 noses[i].x = frame.cols - noses[i].width;
   //             if (noses[i].y + noses[i].height > frame.rows)
   //                 noses[i].y = frame.rows - noses[i].height;
   //         }

   //         rectangle(frame, faces[0], cv::Scalar(0,255,0)); // outline the face
   //if (noses.size() > 0)
   //{
   //	rectangle(frame, noses[0], cv::Scalar(255, 0, 0)); // outline the nose
   //	cv::Point mousePos(0, 0);
   //	faceNoseToMousePos(cv::Size(frame.cols, frame.rows), faces[0], noses[0], mousePos);
   //}
   //     }
   */

		//
		// Display
		//


		// Draw mouse position simulator
		cv::circle(frame, lastMousePos, 10, cv::Scalar(0, 0, 255), 3);

		// Draw status text
		std::string doEqHistStatus = doEqualizeHist ? "equalize hist = on" : "equalize hist = off";
		cv::putText(frame, doEqHistStatus, cv::Point(10, 10), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255));
		std::string minNeighbors_nose_status = "minimum nose neighbors = " + std::to_string(minNeighbors_nose);
		cv::putText(frame, minNeighbors_nose_status, cv::Point(10, 30), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255));
		std::string minNeighbors_face_status = "minimum face neighbors = " + std::to_string(minNeighbors_face);
		cv::putText(frame, minNeighbors_face_status, cv::Point(10, 50), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255));
		std::string claheClipLimitStatus = "CLAHE clip limit = " + std::to_string(claheClipLimit);
		cv::putText(frame, claheClipLimitStatus, cv::Point(10, 70), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255));
		std::string claheTileGridSizeStatus = "CLAHE tile grid size = " + std::to_string(claheTileGridSize.width);
		cv::putText(frame, claheTileGridSizeStatus, cv::Point(10, 90), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255));
		std::string spacebarMsg = "Hit spacebar to recalibrate";
		if (calibrationFrameNum < numFramesToCalibrateNoseRangeWith){
			spacebarMsg = "Calibrating pointer range...";
		}
		cv::putText(frame, spacebarMsg, cv::Point(10, 110), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 100, 255));

		cv::imshow("display", frame);   // this window displays the full-res color frame with pointer etc. overlaid
		//cv::imshow("glareMask", glareMask);
		cv::imshow("preprocessed", smallImg);  // this window displays the preprocessed frame used for detection and tracking

		int key = cv::waitKey(10);
		//std::cout << "key : " << key << std::endl;

		// Key codes for toggling parameters
		const int key_space = 32;
		const int key_c = 99;
		const int key_e = 101;
		const int key_f = 102;
		const int key_g = 103;
		const int key_n = 110;
		if (key > -1) {
			if ((key <= 57) && (key >= 49))
			{
				glareInpaintRadius = 1.0 * (key - 48);
			}
			switch (key) {
			case key_space:
				calibrationFrameNum = 0;
				haveStableChosenPoint = 0;
				break;
			case key_c:
				if (claheClipLimit > 5) claheClipLimit = 1;
				else claheClipLimit++;
				break;
			case key_e:
				doEqualizeHist = doEqualizeHist ? false : true;
				break;
			case key_f:
				if (minNeighbors_face > 10) minNeighbors_face = 2;
				else minNeighbors_face++;
				break;
			case key_g:
				if (claheTileGridSize.width > 20) claheTileGridSize = cv::Size(2, 2);
				else { claheTileGridSize.width++; claheTileGridSize.height++; }
				break;
			case key_n:
				if (minNeighbors_nose > 5) minNeighbors_nose = 1;
				else minNeighbors_nose++;
				break;
			default:
				break;
			}
		}

		frameIdx++;
	}

	return 1;
}
