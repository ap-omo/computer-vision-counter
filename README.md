# Computer Vision Assignment - Black Cat Counter
Development of a computer vision system for counting objects of a certain colour, in this case black cats, in both image and video formats. The program was built using the provided YOLOe .ONNX segmentation model and has the ability to detect cats in each frame, determine each cat's colour using the model's segmentation masks, as well as tracking detected cats across frames in order to avoid duplicate counts, even with abrupt movement and inconsistencies in detection present.

# Requirements
Python 3.12.11 64-bit

opencv-contrib-python-5.0.0.93

Ran on the latest version of Windows 10 using Spyder IDE 6.1.5
