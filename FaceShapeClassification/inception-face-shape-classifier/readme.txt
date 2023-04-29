이 리포지토리에는 사람 얼굴 이미지를 하트, 타원, 타원, 원형 및 사각의 5가지 기본 모양으로 분류하기 위해 Inception v3 모델을 재교육하는 데 사용되는 스크립트가 포함되어 있습니다. 또한 저장소에는 OpenCV 및 DLIB 사전 훈련 모델을 사용하여 생성된 얼굴 랜드마크 좌표에서 파생된 기능을 사용하여 기존 분류기로 벤치마킹하는 데 사용되는 스크립트가 포함되어 있습니다.

사용된 교육용 이미지는 Google 이미지 검색을 통해 다운로드되며 저작권 제약이 있을 수 있습니다. 재배포하지 않고 학술 목적으로만 사용할 것을 약속하면 요청에 따라 이미지, 병목 현상 파일 및/또는 기능 파일의 복사본을 제공할 수 있습니다. 관심이 있으시다면 adonis@eee.upd.edu.ph 으로 메일을 보내주시면 됩니다.

Classify_FACE.PY
이 스크립트는 재교육된 Inception 모델을 실행하여 단일 이미지 또는 이미지 배치를 분류합니다

CLASSIC_FACE_CONFUSION.PY
classify_face.py와 비슷하지만 결과와 혼동 행렬의 텍스트 파일을 생성합니다

EXTRACT_FEATURES.PY
이 스크립트는 이미지에서 얼굴을 감지하고 경계 상자를 지정하며 얼굴 랜드마크를 감지하고 훈련을 위해 형상을 추출합니다

PROCESS_IMAGE.PY
이 스크립트에는 이미지 제곱, 필터, 블러, 줌, 회전, 플립, 리코더 등과 같은 이미지 사전 처리 및 확대 기능이 포함되어 있습니다

RETRAIN_CMDGEN.PY
이 스크립트는 윈도우즈 CMD 명령을 생성하여 CMD 라인 프롬프트를 티잉하는 Inception v3 모델을 텍스트 파일로 재교육합니다. 필요한 파일 및 디렉토리를 설정한 다음 CMD 라인에서 실행하여 모델을 재교육합니다.

RETRAIN_v2.PY
#모든 이미지를 포함하도록 전체 테스트 세트를 정의할 때 약간의 수정, 검증 및 테스트 이미지의 "더블 카운팅" 문제 해결, 유용한 정보를 txt 파일로 저장할 위치에 대한 CMD 라인 인수 추가

TRAIN_CLASSIFIES.PY
이 스크립트는 일련의 교육 세트 크기에 대해 LDA, SVM-LIN, SVM-RBF, MLP 및 KNN 분류기를 교육합니다

종이.PDF
방법론과 실험 결과를 설명하는 짧은 논문

병목 현상
데이터 세트에 있는 500개 이미지의 병목 현상 파일(TXT)을 포함합니다. 참고: 병목 파일은 Inception 모델의 마지막 레이어 이전 이미지의 벡터 표현입니다.

특징들.txt
LDA, SVM, KNN 및 MLP 분류기에 사용되는 데이터 세트의 500개 이미지의 특징 벡터를 포함합니다.

This repository contains the scripts used in retraining the Inception v3 model to classify images of human faces into five basic shapes: heart, oblong, oval, round, and square. The repository also contains the scripts used to benchmark it to traditional classifiers using features derived from facial landmark coordinates generated using OpenCV and DLIB pre-trained models.

Training images used are downloaded via Google image search and may have copyright constraints; I can give you a copy of the images, bottleneck files, and/or feature files by request if you promise not to redistribute and to only use for academic purposes. You can send me an e-mail at adonis@eee.upd.edu.ph if you are interested.

CLASSIFY_FACE.PY
This script runs the re-trained Inception model to classify a single or a batch of images

CLASSIFY_FACE_CONFUSION.PY
Similar to classify_face.py but generates a text file of results and a confusion matrix

EXTRACT_FEATURES.PY
This script detects the face(s) in the image, specifies the bounding box, detects the facial landmarks, and extracts the features for training

PROCESS_IMAGE.PY
This script contains a couple of image pre-processing and augmentation functions like squaring an image, filters, blurs, zoom, rotate, flip, and recolor, etc

RETRAIN_CMDGEN.PY
This script generates the Windows CMD command to re-train the Inception v3 model that tees CMD line prompts into a text file; Set up the needed files and directories then run in the CMD line to retrain the model.

RETRAIN_v2.PY
#Slight modifications in defining the overall test set to include all images, resolved issue of "doubling-counting" of validation and test images, added CMD line arguments on where to save useful info as txt file

TRAIN_CLASSIFIERS.PY
This script trains the LDA, SVM-LIN, SVM-RBF, MLP, and KNN classifiers for a set of training set sizes

PAPER.PDF
A short paper describing the methodology and experimental results

bottlenecks.rar
COntains the bottleneck files (TXT) of the 500 images in the dataset. Note: Bottleneck files are the vector representation of the images at before the last layer of the Inception model.

features.txt
COntains the feature vectors of the 500 images in the dataset used in the LDA, SVM, KNN, and MLP classifiers.
