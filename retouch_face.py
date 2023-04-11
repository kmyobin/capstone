import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageGrab
from pytesseract import *


def get_files_count(folder_path): # 해당 폴더의 파일 개수(폴더 포함) 반환
    dirListing=os.listdir(folder_path)
    return len(dirListing)

def image_trim(img, x, y, w, h):
   img_trim=img[y:y+h, x:x+w]
   return img_trim


length = get_files_count('real_dataset') - 6 # 체커보드 사진 제외하고 데이터셋 개수 세기
    

'''
피부톤 사진(일반광)에서 영상과 숫자를 추출하여 담는다
'''
for i in range(1,length+1):
  file_name="real_dataset/"+str(i)+"/"+str(i)+"_피부톤.bmp"

  img=cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR) # 한글 경로 못읽으므로 dtype 지정
  #print(img)

  img_trim=image_trim(img, 103, 191, 499, 749) # 얼굴 영상만 추출
  num_trim=image_trim(img, 859, 456, 50, 30)
  num_name="num"+str(i)
  #cv2.imshow(num_name, num_trim)
  text=pytesseract.image_to_string(num_trim, config="kor+eng")
  print("text : "+text)
  #cv2.imshow(file_name, img_trim)

  '''
  camera matrix와 distortion coefficients 값을 적절하게 조정
  camera calibration을 통해 camera_matrix, distortion 구함
  '''
  camera_matrix = np.array([[575.440, 0, 182.335],
                            [0, 575.621, 300.01],
                            [0, 0, 1]])
  dist_coeffs = np.array([-0.223114, 0.127922, -0.000141, 0.002677])

  # 왜곡 보정하기
  result_img = cv2.undistort(img_trim, camera_matrix, dist_coeffs)

  #cv2.imshow(name, result_img)    

  save_name="result_dataset/"+str(i)+".bmp" # 저장될 경로
  cv2.imwrite(save_name, result_img)

  print("result"+str(i)+"가 저장되었습니다.")

cv2.waitKey(0)
cv2.destroyAllWindows()
