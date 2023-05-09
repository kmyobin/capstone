import dlib
import cv2
import numpy as np
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
Img_path=os.path.abspath(os.path.join(parent_path,'output','HairStyleChage.jpg'))
output_path=os.path.abspath(os.path.join(parent_path,'output','removeEyebrow.jpg'))

# Load the face landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Create a face detector object
detector = dlib.get_frontal_face_detector()


for i in range(1, 2):
    file_path=Img_path # 얼굴 사진 경로
    img=cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR) # 한글 경로 못읽으므로 dtype 지정
    
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
    
    # 가공한 정보 저장하기
    save_name="remove_eyebrows_dataset/" # 기본 경로
    save_result_img=output_path # 이미지가 저장될 경로

    if not os.path.exists(save_name): # 해당 경로가 존재하지 않는다면
        os.makedirs(save_name)
    
    cv2.imwrite(save_result_img, dst) # 이미지 저장

    print("img.jpg가 저장되었습니다.")
    

    # Display the original and inpainted images side by side
    cv2.imshow('Original', img)
    cv2.imshow('Inpainted', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

