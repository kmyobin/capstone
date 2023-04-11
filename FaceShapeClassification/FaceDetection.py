import cv2
import dlib

# 이미지 경로 지정
img_path = "C:\seeun\Git\capstone\FaceShapeClassification\input\test.jpg"

# 이미지 읽기
img = cv2.imread(img_path)

# dlib의 얼굴 인식기 생성
detector = dlib.get_frontal_face_detector()

# 이미지에서 얼굴 영역 검출
faces = detector(img, 1)

# 검출된 얼굴 영역의 수 만큼 반복
for face in faces:

    # 검출된 얼굴 영역에서 얼굴 특징점 검출
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmarks = predictor(img, face)

    # 검출된 얼굴 특징점 출력
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

# 결과 이미지 출력
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
