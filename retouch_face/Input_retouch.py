import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageGrab
from pytesseract import *

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))

def image_trim(img, x, y, w, h):
   img_trim=img[y:y+h, x:x+w]
   return img_trim


'''
피부톤 사진(일반광)에서 영상과 숫자를 추출하여 담는다
'''
for i in range(1,2):
  file_name=os.path.abspath(os.path.join(parent_path,'MarkVu','피부톤.jpg'))
  file_name_rb=os.path.abspath(os.path.join(parent_path,'MarkVu','red&brown.jpg'))

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
  
  save_name=os.path.abspath(os.path.join(parent_path, 'output')) # 기본 경로
  save_result_img=os.path.abspath(os.path.join(save_name,'retorch_output.jpg')) # 이미지가 저장될 경로

  if not os.path.exists(save_name): # 해당 경로가 존재하지 않는다면
     os.makedirs(save_name)
  
  cv2.imwrite(save_result_img, result_img) # 이미지 저장
  #cv2.imshow('markVu', img)
  #cv2.imshow('markVu', img_rb)
  cv2.imshow('Input : markVu1', cv2.resize(img, (0, 0), fx=0.3, fy=0.3))
  cv2.moveWindow('Input : markVu1', 100, 100)
  cv2.imshow('Input : markVu2', cv2.resize(img_rb, (0, 0), fx=0.3, fy=0.3))
  cv2.moveWindow('Input : markVu2', 500, 100)
  cv2.imshow('output : Wide angle correction image', cv2.resize(result_img, (0, 0), fx=0.4, fy=0.4))
  cv2.moveWindow('output : Wide angle correction image', 100, 400)

  txt_name=os.path.abspath(os.path.join(save_name, 'result.txt')) # 텍스트가 저장될 경로
  with open(txt_name, 'w') as f:
    f.write("피부톤 : " + text +"Red 색소 : " + text_r +"Brown 색소 : " +text_b) # 피부톤, Red 색소, Brown 색소

  print("result"+"가 저장되었습니다.")
  

cv2.waitKey(0)
cv2.destroyAllWindows()