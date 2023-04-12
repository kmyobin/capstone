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
  file_name_rb="real_dataset/"+str(i)+"/"+str(i)+"_R&B.bmp"

  img=cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR) # 한글 경로 못읽으므로 dtype 지정
  img_rb=cv2.imdecode(np.fromfile(file_name_rb, dtype=np.uint8), cv2.IMREAD_COLOR) # 한글 경로 못읽으므로 dtype 지정
  #print(img)

  img_trim=image_trim(img, 103, 191, 499, 749) # 얼굴 영상만 추출
  num_trim=image_trim(img, 859, 456, 50, 30) # 숫자 영상만 추출
  # R&B 색소 평균 추출
  num_trim_r=image_trim(img_rb, 860, 454,  55, 30)
  num_trim_b=image_trim(img_rb, 856, 863,  55, 30)

  scale_percent = 200 # 확대할 비율
  width = int(num_trim_r.shape[1] * scale_percent / 100)
  height = int(num_trim_r.shape[0] * scale_percent / 100)  

  width2 = int(num_trim_b.shape[1] * scale_percent / 100)
  height2 = int(num_trim_b.shape[0] * scale_percent / 100)

  dim=(width, height) 
  dim2=(width2, height2)
  num_trim_r=cv2.resize(num_trim_r, dim, interpolation=cv2.INTER_AREA)
  num_trim_b=cv2.resize(num_trim_b, dim2, interpolation=cv2.INTER_AREA)

  custom_config=r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
  text=pytesseract.image_to_string(num_trim, config="kor")
  text_r = pytesseract.image_to_string(num_trim_r, config=custom_config)
                                     #config='--psm 11 -c tessedit_char_whitelist=0123456789')
  text_b = pytesseract.image_to_string(num_trim_b, config=custom_config)
                                     #config='--psm 11 -c tessedit_char_whitelist=0123456789')

  print("text : " + text)
  print("text_r : " + text_r)
  print("text_b : " + text_b)

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
  
  # 가공한 정보 저장하기
  save_name="result_dataset/"+str(i)+"/" # 기본 경로
  save_result_img=save_name+str(i)+".jpg" # 이미지가 저장될 경로

  if not os.path.exists(save_name): # 해당 경로가 존재하지 않는다면
     os.makedirs(save_name)
  
  cv2.imwrite(save_result_img, result_img) # 이미지 저장

  txt_name=save_name+"result.txt" # 텍스트가 저장될 경로
  with open(txt_name, 'w') as f:
    f.write(text + text_r +text_b) # 피부톤, Red 색소, Brown 색소

  print("result"+str(i)+"가 저장되었습니다.")

cv2.waitKey(0)
cv2.destroyAllWindows()
