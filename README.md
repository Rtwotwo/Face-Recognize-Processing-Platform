# Face-Recognize-Processing-Platform

This software is mainly used to bring facial recognition feature point recognition, face alignment, image enhancement and other algorithms to users in a visual way. At the same time, it adds functions such as live detection, feature point extraction, facial recognition, and image cropping. The main Python facial recognition features and image processing third-party libraries used include dlib, opencv-python, mediapipe, facenet-pytorch, etc.

## 1.Environmental Preparation

Python 3.X and the following libraries are required:

| name          | version    | name           | version    |
|----------------|-----------|----------------|-----------|
| numpy          | 124.3     | dlib           | 19.24.6   |
| opencv-python  | 4.9.0.80  | facenet-pytorch| 2.6.0     |
| pillow         | 10.2.0    | mediapipe      | 0.10.14   |
| imutils        | 0.5.4     | pandas         | 2.2.2     |

## 2.fundamental functions

Now, let's introduce the main functions of this software:
<<<<<<< HEAD
This program is a Python-based face recognition system that uses various computer vision libraries (such as OpenCV, dlib, MediaPipe, facenet_pytorch, etc.) to implement functions such as face detection, keypoint detection, face alignment, liveness detection, face feature extraction, and face recognition. The program builds a Graphical User Interface (GUI) through the Tkinter library, and users can click buttons to turn on or off different functional modules.  
=======
This program is a Python-based face recognition system that uses various computer vision libraries (such as OpenCV, dlib, MediaPipe, facenet_pytorch, etc.) to implement functions such as face detection, keypoint detection, face alignment, liveness detection, face feature extraction, and face recognition. The program builds a Graphical User Interface (GUI) through the Tkinter library, and users can click buttons to turn on or off different functional modules.
>>>>>>> 6e82081e05f34dea288590db48ee365feaa5094f
[演示视频](https://b23.tv/n5PKEK6)

## 3. Usage

|         function         |                             usage                             |
|--------------------------|---------------------------------------------------------------|
|       Face Detection     |  Use the MTCNN model to detect faces in video frames and draw bounding boxes around the detected faces |
|       Keypoint Detection |  Use the dlib library to detect keypoints of the face and draw these keypoints on the video frame |
|       Face Alignment     |  Use the MediaPipe library to detect keypoints of the face, calculate the tilt angle of the face based on these keypoints, and then align and crop the face |
|       Liveness Detection |  Judge whether it is a living person by detecting the blinking of the eyes, the opening and closing of the mouth, and the rotation of the head   |
|  Face Feature Extraction |  Use the MTCNN and InceptionResnetV1 models to extract the features of the face and save these features to a CSV file|
|  Face Recognition        |  Perform face recognition by comparing the face features in the current frame with the face features in the database|
