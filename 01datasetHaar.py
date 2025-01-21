import cv2
import numpy as np
import os

# Load Haar Cascade model for face detection
# This loads the pre-trained Haar Cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Initialize video capture (0 = default camera, usually the webcam)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set video width to 640 pixels
cap.set(4, 480)  # Set video height to 480 pixels

# Prompt the user to enter a unique user ID and name
user_id = input('\nEnter user ID and press <return> ==> ')
user_name = input('\nMake sure only you in front of camera\nEnter user name and press <return> ==> ')

# Set the directory to save the dataset if it doesn't exist
dataset_dir = 'datasetcolor'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)  # Create the directory if it doesn't exist

# Create a unique folder for each user using their ID and name
user_folder_color = os.path.join(dataset_dir, f"{user_id}_{user_name}")
if not os.path.exists(user_folder_color):
    os.makedirs(user_folder_color)  # Create the user's folder if it doesn't exist

print("\n[INFO] Initializing face capture. Look at the camera and wait...")

# Initialize a count variable to keep track of the number of images captured
count = 0

# Define minimum window size to be recognized as a face, 10% of the video frame size
minW = 0.1 * cap.get(3)
minH = 0.1 * cap.get(4)

while True:
    # Capture frame-by-frame from the webcam
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame.")  # Exit if the frame is not captured properly
        break

    # Flip the image horizontally to create a mirror-like effect (optional)
    img = cv2.flip(img, 1)

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    # scaleFactor=1.3 scales the image to detect faces at different sizes
    # minNeighbors=5 specifies how many neighbors each rectangle should have to retain it
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face in the color image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle

        # Increment the face sample count
        count += 1

        # Extract the face region from the color image and save it
        face_roicolor = img[y:y + h, x:x + w]
        face_filenamecolor = os.path.join(user_folder_color, f"User.{user_id}.{count}.jpg")
        cv2.imwrite(face_filenamecolor, face_roicolor)  # Save the image to the specified directory
        print(f"Face image saved: {face_filenamecolor}")

    # Display the video frame with detected faces
    cv2.imshow('Image', img)

    # Wait for a key press: 
    # Press 'ESC' (key code 27) to exit the video loop
    # Or stop after capturing 30 face samples
    k = cv2.waitKey(100) & 0xff  # 100ms delay between frames
    if k == 27:  # Exit if 'ESC' is pressed
        break
    elif count >= 30:  # Stop after taking 30 face samples
        break

# Cleanup the resources after capturing is done
print("\n[INFO] Exiting Program and cleaning up...")
cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
