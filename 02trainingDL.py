from imutils import paths
import face_recognition
import pickle
import cv2
import os

# Print a message indicating that the face encoding process is starting
print("[INFO] start processing faces...")

# Get the list of image file paths from the dataset directory (e.g., 'datasetcolor')
imagePaths = list(paths.list_images("datasetcolor"))

# Initialize two lists: one for storing face encodings and another for corresponding names
knownEncodings = []
knownNames = []

# Loop over each image path
for (i, imagePath) in enumerate(imagePaths):
    # Print the current progress (e.g., which image is being processed)
    print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))

    # Extract the name of the person from the directory name (parent folder)
    name = imagePath.split(os.path.sep)[-2]

    # Load the input image and convert it from BGR (OpenCV default) to RGB (required by face_recognition library)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the face encodings (128-dimension face embeddings) for each face in the image
    encodings = face_recognition.face_encodings(rgb)

    # Loop over each detected face encoding
    for encoding in encodings:
        # Append the face encoding and the associated name to the respective lists
        knownEncodings.append(encoding)
        knownNames.append(name)

# After all images are processed, print a message that the encodings are being serialized (saved to disk)
print("[INFO] serializing encodings...")

# Create a dictionary to store the encodings and corresponding names
data = {"encodings": knownEncodings, "names": knownNames}

# Open a file in write-binary mode ('wb') and save the encoded face data using pickle
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))  # Serialize the data into the file

# Wait for user input before exiting the program
input("Press enter to exit...")

