# AI_FaceRecognition
The AI Face Recognition Program consists of three distinct code modules designed to collect datasets, train a machine learning model, and perform face recognition.

---
## **Run the Program**
To run the program, you only need to execute the code sequentially. First, run "01datasetHaar.py", then "02trainingDL.py", and finally "03recognitionDnnDL.py".

---

## **Explanation**
### **1. Dataset Collection Program**
The first code is a program to gather dataset of known faces. The code makes use of Haar Cascade 
Classifier machine learning object detection method to detect frontal face. When a user starts the 
program and faces the camera, the program will detect the user’s face, capture multiple pictures, crop 
the faces, and store the cropped frontal faces in a dataset folder. The user can restart the program to 
capture another person's dataset. At the end of it, we will have a dataset folder consisting of multiple 
folders of known persons’ images.

**Outcome:**
A dataset folder containing subfolders of cropped frontal face images for each known person.

### **2. Model Training Program**
The second code is a program to train a machine learning model from face_recognition library with 
the gathered image dataset of known persons. The machine learning model used is a Deep 
Convolutional Neural Network (CNN) based on the ResNet architecture that was trained on a large 
dataset to extract facial features. This model will generate 128-dimensional face encodings from the 
images of our dataset, then, a pickle file of these encodings will be generated. This encoding file will 
be used in the third code for recognition purposes.

**Outcome:**
A pickle file containing the 128-dimensional encodings of known faces.

### **3. Face Detection and Recognition Program**
The third code is a program to detect faces and to recognize them. The detecting process is done 
through DNN face detection model: ResNet-10, which is a trained model file based on the Single Shot 
Multibox Detector (SSD) framework, specifically designed for detecting faces. After faces are 
detected, the faces will be labelled based on recognition process. This recognition is done by the face 
recognition Deep CNN model (the same model as in the second code). Essentially, once the face 
locations are detected by DNN, the Deep CNN model computes 128-dimensional face encodings of 
the detected faces. Then, these embeddings are compared with the previously stored encodings (of 
known faces from the second code) to identify the person.

**Outcome:**
Detected faces are labeled with the corresponding person's identity.






