import dlib
import cv2
import numpy as np

# Load the face landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Create a face detector object
detector = dlib.get_frontal_face_detector()

# Load the input image
img = cv2.imread('4.jpg')

# Convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the faces in the grayscale image
faces = detector(gray)

# Loop over each face
for face in faces:
    # Detect the facial landmarks
    landmarks = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    # Extract the coordinates of the eyebrows
    left_eyebrow = landmarks[17:22]
    right_eyebrow = landmarks[22:27]
    
    # Create a mask for the eyebrows
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [left_eyebrow], 0, 255, -1)
    cv2.drawContours(mask, [right_eyebrow], 0, 255, -1)

    # Apply Gaussian blur to the mask
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    # Apply image inpainting to fill the eyebrow region
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

# Display the original and inpainted images side by side
cv2.imshow('Original', img)
cv2.imshow('Inpainted', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()