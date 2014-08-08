# FacePointer
Uses a webcam and facial feature tracking to move an onscreen pointer.  Intended as a prototype to be extended for control of a UI.  

Developed in discussion with Dan Bacher of the Speak Your Mind Foundation, http://speakyourmindfoundation.org

## Requirements
* OpenCV 2.4 (maybe 2.4.9)
* CMake 2.8

## Build instructions
### Mac / linux
Install OpenCV and CMake.  Then ccmake and make and it should work (tested on OSX 10.9)
### Windows
Install OpenCV, CMake, and Visual Studio.  Use CMake to generate a VS project, and build.

## Usage Notes
* The first few frames of successful face detection are used for range calibration, so don't forget to trace some circles with your nose until it indicates that calibration is complete.  This usually takes 10-20 seconds.  Once that's calibrated, optical flow tracking kicks in for a more resilient and efficient tracking.
* The default preprocessing settings were robust across a few daylight and evening scenes in my testing.  Toggling through the settings can help in certain conditions; this would ideally be automated, but for now, keys for toggling are specified at the bottom of the main source file.
