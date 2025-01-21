# AI_FaceRecognition
The AI Face Recognition Program consists of three distinct code modules designed to collect datasets, train a machine learning model, and perform face recognition.

---

## **1. Dataset Collection Program**

This module is responsible for collecting a dataset of known faces using the Haar Cascade Classifier, a machine learning object detection method specialized in detecting frontal faces.

**Process:**

The user starts the program and faces the camera.

The program detects the user's face.

Multiple pictures are captured.

Detected faces are cropped and saved in a dataset folder.

The process can be repeated to collect datasets for multiple individuals.

**Outcome:**

A dataset folder containing subfolders of cropped frontal face images for each known person.


**2. Model Training Program**

This module utilizes the face_recognition library to train a Deep Convolutional Neural Network (CNN) based on the ResNet architecture. This model has been pre-trained on a large dataset to extract facial features.

**Process:**

Load images from the dataset folder.

Generate 128-dimensional face encodings using the pre-trained Deep CNN model.

Store the encodings in a pickle file for future use in the recognition process.

**Outcome:**

A pickle file containing the 128-dimensional encodings of known faces.


**3. Face Detection and Recognition Program**

This module detects and recognizes faces using a combination of face detection and recognition methods.

**Process:**

**Face Detection:**

Utilizes the DNN face detection model, ResNet-10, which is based on the Single Shot Multibox Detector (SSD) framework, optimized for face detection.

Detects face locations within the camera feed.

**Face Recognition:**

The Deep CNN model computes 128-dimensional face encodings of detected faces.

These encodings are compared with stored encodings from the training phase to identify the individual.

**Outcome:**

Detected faces are labeled with the corresponding person's identity.

---

### **Run the Program**

To run the program, you only need to execute the code sequentially. First, run "01datasetHaar.py", then "02trainingDL.py", and finally "03recognitionDnnDL.py".
