# Object-Detection

This project is designed to print the class of objects in a photo or video and the frame number that 
objects are belong to. YOLOv4 is used as pretrained object detection model. The functions can be runned by using terminal arguments. 

# To run, please follow these steps:

First, download the YOLOv4 weights file (245 Mb) from the link and locate it to " pretrained_model " directory:
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

#For video example, run this code in your terminal:

>> python video.py --input videolar/school_video.mp4 


#For photo example:


>> python fotograf.py --input fotograflar/school.jpg


Please make sure that your terminal is opened in your project directory. Input files can be customised by adding 
their specified directories. 

