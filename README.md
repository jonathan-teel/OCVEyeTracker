# OCVEyeTracker.cpp using OpenCV2


## Webcam eye movement tracker that customizes to the user's view pattern with a feedforward neural network

### Includes classes 
Perceptron.cpp
Timer.cpp
TrainingEnv.cpp
	
--------------------

Use the following command for the required packages :

sudo apt-get install build-essential libgtk2.0-dev libjpeg-dev  libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen2-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev


--------------------

**This detects the face and eyes, then splits each eye into 4 quadrants centered around the pupil**. The black/white shading of each of these quadrants is tracked and used fed to the feedforward algorithm to determine which quadrant the pupil is in.

**The original intention was for the code to be used with onboarded systems, that need to adjust when the person couldn't. For example, a quadriplegic could have software customized that would simulate keyboard and mouse functions.** No future updates are planned at this time.


	
