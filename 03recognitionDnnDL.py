from imutils.video import VideoStream  # For video capture from a webcam or video source
from imutils.video import FPS          # To measure and display frames per second (FPS)
import face_recognition                # Face recognition library for detecting and recognizing faces
import imutils                         # Helper functions for resizing and manipulating images
import pickle                          # For loading and saving data in binary format
import time                            # Used for adding delays (e.g., warming up the camera)
import cv2                             # OpenCV library for image processing
import numpy as np                     # NumPy for array manipulation, especially for handling image data

# Variable to keep track of the last recognized name
currentname = "unknown"

# Path to the pickle file where face encodings are stored
encodingsP = "encodings.pickle"

# Load the pre-trained face detection model using DNN (Deep Neural Network)
protoPath = "deploy.prototxt"  # Path to the prototxt file defining the model architecture
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the pre-trained model weights
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)  # Load the model into the DNN framework

# Load face encodings that were previously saved in the pickle file
print("[INFO] loading encodings...")
data = pickle.loads(open(encodingsP, "rb").read())  # Load the serialized face encodings

# Start video stream using the webcam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()  # src=0 means the default webcam will be used
time.sleep(2.0)  # Wait for 2 seconds to allow the camera to warm up

# Start the FPS counter to track the frame rate
fps = FPS().start()

# Loop to continuously capture frames from the webcam and process them
while True:
    frame = vs.read()  # Read the current frame from the video stream
    frame = imutils.resize(frame, width=500)  # Resize the frame to a width of 500px for faster processing

    # Flip the frame horizontally to create a mirror effect (optional for better user experience)
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    (h, w) = frame.shape[:2]

    
    #####################################################################
    #--------Code for face detection using CAFFE DNN--------------------#

    # Pre-process the frame for face detection by resizing and normalizing
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))  # Convert the frame to a blob for DNN processing
    net.setInput(blob)  # Feed the blob into the network
    detections = net.forward()  # Perform face detection

    # Initialize an empty list to store face bounding boxes
    boxes = []

    # Loop through the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Extract confidence score for each detection

        # Only consider detections with confidence greater than 0.5
        if confidence > 0.5:
            # Compute the (x, y) coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxes.append((startY, endX, endY, startX))  # Append bounding box coordinates to the list
    #--------End of face detection using CAFFE DNN--------------------#

    ##########################################################################
    #--Code for face recognition using Deep CNN (face recognition library)---#

    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Compute the facial encodings (128-dimension feature vector) for each face in the frame
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Initialize an empty list to store the names of recognized individuals
    names = []

    # Loop over the facial encodings
    for encoding in encodings:
        # Compare each face encoding to the known encodings in the pickle file
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"  # Default name if no match is found

        # If there is a match
        if True in matches:
            # Get the indices of matched encodings
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Count the occurrences of each matched name
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Determine the name with the highest count (most matches)
            name = max(counts, key=counts.get)

            # If the current name is different from the last recognized name, print the new name
            if currentname != name:
                currentname = name
                print(currentname)

        # Append the recognized name (or "Unknown") to the list of names
        names.append(name)
    
    #--End of face recognition using Deep CNN (face recognition library)-----#

    ##########################################################################
    #--Code to draw bounding boxes and label faces---------------------------#

    # Loop over the recognized faces and their corresponding names
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Draw the bounding box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Determine the position for the name label
        y = top - 15 if top - 15 > 15 else top + 15

        # Draw the name of the person below the bounding box
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 2)

    #--End of code to draw bounding boxes and label faces-------------------#

    # Display the resulting frame with the bounding boxes and names
    cv2.imshow("Facial Recognition is Running", frame)

    # Break the loop if the 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the FPS counter and print the elapsed time and FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup: close all windows and stop the video stream
cv2.destroyAllWindows()
vs.stop()
