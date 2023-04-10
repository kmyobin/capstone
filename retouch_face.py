import cv2
import numpy as np

img=cv2.imread('dataset/ex.png', cv2.IMREAD_UNCHANGED) # 절대경로로 진행할 것 
#print(img)
cv2.imshow("ex", img)


# camera matrix와 distortion coefficients 값을 적절하게 조정
camera_matrix = np.array([[575.440, 0, 182.335],
                          [0, 575.621, 300.01],
                          [0, 0, 1]])
dist_coeffs = np.array([-0.223114, 0.127922, -0.000141, 0.002677])

# 왜곡 보정하기
img = cv2.undistort(img, camera_matrix, dist_coeffs)

cv2.imshow("result", img)

cv2.waitKey(0)
cv2.destroyAllWindows()