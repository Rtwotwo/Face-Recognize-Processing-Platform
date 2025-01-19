# Face-Recognize-Processing-Platform

This software is mainly used to bring facial recognition feature point recognition, face alignment, image enhancement and other algorithms to users in a visual way. At the same time, it adds functions such as live detection, feature point extraction, facial recognition, and image cropping. The main Python facial recognition features and image processing third-party libraries used include dlib, opencv-python, mediapipe, facenet-pytorch, etc.

## 1.Environmental Preparation

Python 3.X and the following libraries are required:

| name          | version    | name           | version   |
|----------------|-----------|----------------|-----------|
| numpy          | 124.3     | dlib           | 19.24.6   |
| opencv-python  | 4.9.0.80  | facenet-pytorch| 2.6.0     |
| pillow         | 10.2.0    | mediapipe      | 0.10.14   |
| imutils        | 0.5.4     | pandas         | 2.2.2     |

## 2.Video Introduction

Now, let's introduce the main functions of this software:  
This program is a Python-based face recognition system that uses various computer vision libraries (such as OpenCV, dlib, MediaPipe, facenet_pytorch, etc.) to implement functions such as face detection, keypoint detection, face alignment, liveness detection, face feature extraction, and face recognition. The program builds a Graphical User Interface (GUI) through the Tkinter library, and users can click buttons to turn on or off different functional modules [演示视频](https://b23.tv/n5PKEK6) or BiliBili: [演示视频](asserts/程序演示视频.mp4).

## 3.Usage

|         function         |                             usage                             |
|--------------------------|---------------------------------------------------------------|
|       Face Detection     |  Use the MTCNN model to detect faces in video frames and draw bounding boxes around the detected faces |
|       Keypoint Detection |  Use the dlib library to detect keypoints of the face and draw these keypoints on the video frame |
|       Face Alignment     |  Use the MediaPipe library to detect keypoints of the face, calculate the tilt angle of the face based on these keypoints, and then align and crop the face |
|       Liveness Detection |  Judge whether it is a living person by detecting the blinking of the eyes, the opening and closing of the mouth, and the rotation of the head   |
|  Face Feature Extraction |  Use the MTCNN and InceptionResnetV1 models to extract the features of the face and save these features to a CSV file|
|  Face Recognition        |  Perform face recognition by comparing the face features in the current frame with the face features in the database|

## 4.Fundamental Function

You can use the following commands to run the main program:

```bash
# Run the main program to experience multiple functions
python face_recognizer_gui.py
```

___feature point extraction___
Feature point extraction is based on the MTCNN deep learning model to extract key point data of human faces and save it in a csv file. Compared with the traditional dlib for extracting feature points mentioned above, the MTCNN model has the advantage that this method has a stronger anti-interference ability when extracting facial features and will not encounter situations where the face is tilted or the background is too dark to lock the feature points.
[特征点提取](asserts/特征点提取.png)
[特征点提取](asserts/特征点提取_流程.png)

___face detection___
Face detection is a fundamental task in computer vision, aiming to locate the positions of human faces in images or videos. Usually, face detection algorithms output a rectangular box (Bounding Box) indicating the detected face area.
Meanwhile, let's introduce the MTCNN (Multi-task Cascaded Convolutional Networks) used: MTCNN is a multi-task cascaded convolutional neural network that can simultaneously complete face detection and face key point localization. It gradually refines the detection results through three cascaded networks (P-Net, R-Net, O-Net), and finally outputs the face box and key point coordinates.
[人脸检测](asserts/人脸检测.png)
[人脸检测](asserts/人脸检测_流程.png)

___key point detection___
Key point detection is an important task in computer vision, aiming to locate the key feature points (such as eyes, nose, mouth, etc.) on a human face. This system uses the 68-point face key point detection model provided by the dlib library. This model is based on HOG (Histogram of Oriented Gradients) features and linear classifiers, and can accurately detect 68 key points of a human face.
[关键点检测](asserts/关键点检测.png)
[关键点检测](asserts/关键点检测_流程.png)

___face alignment___
The main goal of the alignment display function is to detect human faces in the video stream and align them based on the key points of the face (such as eyes, nose, mouth, etc.), ensuring that the human face maintains a frontal posture in the image. This function detects the key points of the face, calculates the tilt angle of the face, and rotates the image to keep the human face horizontally aligned in the image.
Considering that dlib only extracts features for frontal faces and its performance is rather sensitive to light changes; meanwhile, the coordinates extracted by MTCNN are still unknown. For the above reasons, mediapipe is used to extract more accurate facial features.
[对齐显示](asserts/对齐显示.png)
[对齐显示](asserts/对齐显示_流程.png)

___liveness detection___
Specifically, we determine the specific positions of the feature points through the mesh grid of mediapipe, thereby achieving the calculation of the ratios of blinking, head turning, and mouth opening and closing. For the specific blink detection, by calculating the distance of the key points of the eyes, the aspect ratio of the eyes (EAR) is obtained. When the EAR is lower than a certain threshold, the eyes are considered closed. For the mouth opening detection, by calculating the distance of the key points of the mouth, the aspect ratio of the mouth (MAR) is obtained. When the MAR is higher than a certain threshold, the mouth is considered open. For the head turning detection, by calculating the ratio change of the horizontal and vertical lengths of the key points of the face, it is determined whether the head is turned.
[活体检测](asserts/活体检测.png)
[活体检测](asserts/活体检测_流程.png)

___face recognition___
The main goal of the face recognition function is to detect and identify the human faces in the frames captured by the camera through the video. This function extracts the face features of the current stage and compares them with the face features in the database to identify the identity of the face. This function combines two methods, MTCNN and InceptionResnetV1 model, for implementation and can effectively handle the face recognition problem in the video stream.
[人脸识别](asserts/人脸识别.png)
[人脸识别](asserts/人脸识别_流程.png)

## 5.Thanks

Finally, Thank you for watching. Please could you light up a little star.
