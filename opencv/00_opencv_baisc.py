# 파이썬 OpenCV 프로그래밍 입문과 활용

# # 컴퓨터 비전이란?

# high-level undrstanding
# 어떤 영상 또는 이미지를 봤을 때 이게 무엇인지 여기 안에 뭐가 있는지를
# 검출 또는 인식하는 등의 하이 레벨 언더스탠딩하는 것을 컴퓨터 비전이라고 한다
# 엔지니어링 관점에서는 사람의 visual system, 시각 체계를 컴퓨터가 똑같이 모사할 수 있게끔 하는 작업을 컴퓨터 비전이라고 한다

# 컴퓨터비전은 상당히 어렵고 알고리즘도 많이 필요하고 시간도 많이 필요하다
# 사과 사진에서 사과를 인식하는 것은 우리에게 쉽다
# 그렇지만 컴퓨터에게 사과를 인식하도록 어떤 알고리즘을 짜게 하고 싶을 때...
# 둥글다, 빨간색이다 등의 정보를 줄 수 있을 것이다
# 하지만 작은 부분을 차지하지만, 사과의 꼭지는 둥글다는 정보에 어긋난다
# 빨강색이 주로 이루지만 완전히 모두 빨갛지는 않다, 빨강색의 기준은 어디에?
# 기준 설정이 쉽지가 않다

# 토마토와 사과를 어떻게 구분? 어디서 어디부터가 꼭지?
# 사과가 여러 개인 사진에서 빨강색과 비슷한 계열 색이 중구난방으로 보일 때 사과를 몇 개로 인식?


# # 컴퓨터 비전(Computer Vision) vs 영상 처리(Image Processing)

# 어떤 사람들은 컴퓨터 비전 안에 이미지 프로세싱이 포함된다고 생각
# - 컴퓨터 비전이 더 포괄적
# - 이미지 프로세싱을 노이즈 제거 등의 이미지 전처리라고 생각함
# 어떤 사람들은 이미지 프로세싱 안에 컴퓨터 비전이 포함된다고 생각
# - 이미지 프로세싱이 더 포괄적
# - 이미지를 다루는 건 다 이미지 프로세싱, 그 중에 high-level understanding이 들어가는 것이 컴퓨터 비전이라고 생각

# 개념이 명확하진 않음
# 예전만 해도 컴퓨터 비전이란 용어가 잘 쓰이지 않았음
# 그래서 후자로 생각하는 사람들이 많았다

# 하지만 요즘은 컴퓨터 비전이란 말이 흔하게 사용되면서
# 전자로 생각하는 사람들이 많아지고 있다

# 명확히 구분하기는 어렵다 
# 그냥 시대별로 달라졌다, 정도로 보자

# # 컴퓨터 비전 영상처리 관련 분야
# 1. 수학 - 선형대수, 확률/통계, 해석학, 기하학 (아주 중요)
# 2. 신호처리 - 아날로그 신호처리, 디지털 신호처리, 주파수 분석
# 3. 컴퓨터 그래픽스
# 4. 인지 과학
# 5. 컴퓨터 과학 - 수치해석, 알고리즘, 최적화(중요)
# 6. 머신러닝 - 딥러닝, 패턴인식
# 7. 로봇 공학
# 8. 광학

# 그리고 프로그래밍 기법도 중요하다
# 수학적 알고리즘을 프로그램으로 어떻게 잘 만들 것인가...


# # 컴퓨터 비전 연구 분야
# 1. 영상의 화질 개선
# - Filtering App
# - HDR
# - Image Noise Remove
# - Super Resolution(SD 화질을 FHD, 4K까지 복원)

# 2. 객체 검출(Object Detection)
# - 얼굴 검출
# - 보행자 검출

# 3. 분할(Segmentation)
# - 입력 영상을 특정 구역으로 나누거나 필요한 부분만 찾아주는 것

# 4. 인식(Recognition)
# - 얼굴을 검출해서 누구의 얼굴인지까지 인식
# - 문자 인식(OCR)
# - 사물의 위치, 구역 인식(ex.YOLO)

# 5. 움직임 정보 추출 및 추적
# - ex.OpenPOSE, 움직이는 영상에서 사람의 동작을 판단하는 전처리 기술의 하나라고 보면 된다

# # 컴퓨터 비전 응용 분야
# 1. 머신 비전(machine vision)
# - 공장 자동화 : 제품의 불량 검사, 위치 확인, 측정 등
# - 높은 정확도와 빠른 처리 시간 요구
# - 조명, 렌즈, 필터, 실시간(Real-Time)처리

# 2. 인공지능(AI) 서비스
# - 인공지능 로봇과 자율주행 자동차
# - 입력 영상을 객체와 배경으로 분할 -> 객체와 배경 인식 -> 상황 인식 -> 로봇과 자동차의 행동 지시
# - Computer Vision + Sensor Fusion + Deep Learning
# - Amazon Go / 구글, 테슬라의 자율 주행 자동차

# # 영상의 획득
# - 디지털 카메라에서 영상의 획득 과정
# : 피사체 -> 렌즈 -> 센서 -> ISP(영상처리모듈 노이즈 제거, 색상 조정 등) -> 사진파일

# # 영상의 표현 방법
# - 영상(image)이란? 픽셀(pixel)이 바둑판 모양의 격자에 나열되어 있는 형태(2차원 행렬)

# # 영상의 표현 방법
# - 영상에서 사용되는 좌표계
#      0  1  2  ...  w-1
# 0		-> x
# 1
# 2
# .
# .
# .
# h-1 
#      y

# * w x h 영상(w-by-h image)
# 좌표계는 zero-base 왼쪽 위의 점을 0,0으로 인식함
# OpenCV 라이브러리는 영상을 아래 행렬의 형태로 생각함

# A = a1,1 a1,2 ... a1,N
#       a2,1 a2,2 ... a2,N

#      aM,1 aM,2 ... aM,N

# * M x N 행렬(m-by-n matrix)
# 수학 생렬에서는 왼쪽 위가 1,1 이지만 OpenCV에서는 0,0임

# 1920 x 1280에서 1920이 가로 픽셀의 크기, 280이 세로 픽셀의 크기
# 하지만 OpenCV, 행렬을 다룰 때에는 다르다
# 행렬은 행과 열의 조합
# 행의 크기와 열의 크기 순서로 보통 표현을 한다
# M이 행의 갯수, 세로 크기
# N이 열의 갯수, 가로 크기
# 영상을 표현할 때 가로 x 세로
# 행렬을 표현할 때 세로 x 가로
# co-work 시 해상도를 이야기 할 때 주의해야 한다

# - 그레이스케일(grayscale) 영상
# * 색상 정보가 없이 오직 밝기 정보로만 구성된 영상
# * 밝기 정보를 256 단계로 표현(가장 어두움 : 0, 가장 밝음 : 255)

# - 트루컬러(truecolor) 영상
# * 컬러 사진처럼 색상 정보를 가지고 있어서 다양한 색상을 표현할 수 있는 영상
# * Red, Green, Blue 색 성분을 256 단계로 표현, 256^3 = 16,777,216 새개상 표현 가능

# - 픽셀(pixel) 
# * 영상의 기본 단위, picture element, 화소



# # 그레이 스케일 영상과 컬러 영상
# - 그레이스케일 값의 범위
# * 그레이 스케일 영상에서 하나의 픽셀은 0부터 255 사이의 정수 값을 가진다
# * C/C++에서 unsigned char로 표현(1byte)
# typedef unsigned char BYTE; // Windows
# typedef unsigned char uint8_t // Linux
# typedef unsigned char uchar; // OpenCV

# * Pyhon에서는 numpy.uint8

# - 트루컬러 영상에서 하나의 픽셀에 R, G, B 세 가지 값이 들어 있음
# ex) 145(R), 195(G), 228(B) 이게 하나의 픽셀

# opencv의 경우 C언어로 접근할 때에는 픽셀 하나가 세 개의 값을 갖는 방법으로 인식하고
# 파이썬의 경우 3차원 행렬로 표현한다
# blue, green, red 성분을 3가지 축으로 하여 3차원 행렬 형태로 만들어 컬러 영상을 표현한다
# numpy를 이용해서 영상을 다룰 때에 3차원까지 고려를 해야 한다

# # 영상의 표현 방법
# - 영상 데이터 크기 분석
# * 그레이스케일 영상 : 가로 크기 X 세로 크기 Bytes
# ex) 512 * 512 = 262144 Bytes

# * 트루컬러 영상 : 가로 크기 X 세로 크기 X 3 Bytes
# ex) 1920 * 1080 * 3 = 6220800 Bytes(대략 6 MBytes)

 
# <참고> 1초에 보통 30프레임, 많게는 60프레임

# # 영상 파일 형식 특징
# 1. BMP (MS에서 만든 포맷, 영상을 간단하게 저장하기 위해 만듦)
# - 픽셀 데이터를 압축하지 않고 그대로 저장, 파일 용량이 큰 편
# - 파일 구조가 단순해서 별도의 라이브러리 도움 없이 직접 파일 입출력 프로그래밍 가능

# 2. JPG
# - 주로 사진과 같은 컬러 영상을 저장하기 위해 사용
# - 손실 압축(lossy compression)(픽셀값이 미세하게 바뀌는 것 때문에 영상 처리에선 선호하진 않음)
# - 압축률이 좋아서 파일 용량이 크게 감소 -> 디지털 카메라 사진 포맷으로 주로 사용됨

# 3. GIF(영상 처리에선 사실 많이 안 씀)
# - 256 색상 이하의 영상을 저장 -> 일반 사진을 저장 시 화질 열화가 심함
# - 무손실 압축(lossless compression)
# - 움직이는 GIF 지원

# 4. PNG(압축은 하지만 픽셀값이 바뀌진 않음, TIF도 마찬가지)
# - Portable Network Graphics
# - 무손실 압축(True Color 영상도 무손실 압축)
# - 알파 채널(투명도)을 지원

# 영상 처리에선 TIF, PNG 쓰는 것이 좋고 용량이 아주 큰 데이터가 아니라면 BMP를 쓰면 된다


# # OpenCV 개요

# - What is OpenCV?
# * Open Source
# * Computer vision & machine learning
# * Software library

# - Why OpenCV?
# * BSD license... Free for academic & commercial use
# * Multiple interface... C, C++, Python, Java, JavaScript, MATLAB, etc.
# * Multiple platform... Windows, Linux, Mac OS, iOS, Android
# * Optimized... CPU instructions, Multi-core processing, OpenCL, CUDA
# * Usage... Stitching streetview images, detecting intrusions, monitoring mine equipment, helping robots navigate and pick up objects, Interactive art, etc.

# pip install opencv-python==4.1.0.25
# pip install opencv-contrib-python==4.1.0.25

# 현재 최신 버전 4.3 대는 imread() 에서 버그가 있기 때문에
# 4.1.0.25로 설치하는 것이 좋다

# 나는 파이썬이 3.8.6 상태인데 여기서 opencv 해당 버전이 설치가 안 돼서
# 'study'라는 가상환경에서 파이썬 3.7.6 으로 깔고 opencv 설치 진행함


# # OpenCV-Python 코딩 작업 환경
# 1. 메모장 + 명령프롬프트

# 2. 주피터노트북(Jupyter Notebook)
# - 웹브라우저에서 파이썬 코드를 작성 및 실행
# - 블록 단위 코딩
# - 마크업 언어와 그림 등을 활용한 설명 추가가 쉬움

# 3. 파이썬IDE
# - Visual Studio Code, PyCharm, Spider 등
# - OpenCV에서 제공하는 GUI 기능 사용가능
# - 편리한 디버깅

# * IDE : 통합 개발 환경, Integrated Development Environment의 줄임말
# - 공통된 개발자 툴을 하나의 그래픽 사용자 인터페이스(Graphical User Interface)로 결합하는 애플리케이션을 구축하기 위한 소프트웨어










































































