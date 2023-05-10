import os
import cv2
# 파일 경로 리스트

#순서 - 얼굴 광각보정, rgb 추출, R&B 색소 추출, 얼굴형 검출, 눈썹 제거, 헤어 염색
file_paths = ['retouch_face/Input_retouch.py', 'FaceShapeClassification/inception-face-shape-classifier/classify_face.py',
              'Eye_Remove/demo_preprocess.py', 'Eye_Remove/demo_remove.py', 'Eye_Remove/demo_remove_show.py',
              'Eye_Syn/demo_preprocess.py', 'Eye_Syn/demo_synthesis.py', 'Eye_Syn/demo_synthesis_show.py',
              'HairColor_Transfer/makeup.py']

# 파일 실행 
for file_path in file_paths:
    os.system(f'python {file_path}')

with open('output/result.txt', 'r') as f:
    print(f.read())
