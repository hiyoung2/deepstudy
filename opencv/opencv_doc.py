# OpenCV API

# 1. 영상 파일 불러오기
# cv2.imread(filename, flags=None) -> retval
# 1) filename : 불러올 영상 파일 이름(문자열)
# (1) 상대 경로 : 'cat.bmp', '../data/cat.bmp'
# (2) 절대 경로 : 'c:\cat.bmp', '/home/id/cat.bmp'
# 2) flags : 영상 파일 불러오기 옵션 플래그
# (1) cv2.IMREAD_COLOR : BGR 컬러 영상으로 읽기(default), shape = (rows, cols, 3)
# (2) cv2.IMREAD_GRAYSCALE : 그레이스케일 영상으로 읽기, shape = (rows, cols)
# (3) cv2.IMREAD_UNCHANGED : 영상 파일 속성 그대로 읽기(e.g. 투명한 PNG파일 : shape = (rows, cols, 4))
# retval : 불러온 영상 데이터(numpy.ndarray)

# opencv 관련 함수들은 document로 보면 된다 https://docs.opencv.org/master/

# 2. 영상 파일 저장하기
# cv2.imwrite(filename, img, params = None) -> retval
# filename : 저장할 영상 파일 이름(문자열)
# img : 저장할 영상 데이터(numpy.ndarray)
# params : 파일 저장 옵션 지정(속성&값의 정수 쌍)
# e.g. JPG 파일 압축률을 90%로 지정하고 싶다면? [cv2.IMWRITE_JPEG_QUALITY,90] 지정
# retval : 정상적으로 저장하면 True, 실패하면 False

# 3. 새창 띄우기 & 창 닫기
# cv2.namedWindow(winname, flags=None) -> None
# winname : 창 고유 이름, 이 이름으로 창을 구분함
# flags : 창 속성 지정 플래그
# 1) cv2.WINDOW_NORMAL : 영상 크기가 창 크기에 맞게 지정됨
# 2) cv2.WINDOW_AUTOSIZE : 창 크기가 영상 크기에 맞게 자동으로 변경됨(default)

# cv2.destroyWindow(winname) -> None
# cv2.destroyAllWindows() -> None
# winname : 닫고자 하는 창 이름
# 일반적인 경우 프로그램 종료 시 운영 체제에 의해 열려 있는 모든 창이 자동으로 닫힘

# 4. 창 위치 & 크기 지정
# cv2.moveWinidow(winname, x, y) -> None
# winname : 창 이름
# x, y : 이동할 위치 좌표

# cv2.resizeWindow(winname, width, height) -> None
# winname : 창 이름
# width, height : 변경할 창 크기
# 참고 사항
# * 창 생성시 cv2.WINDOW_NORMAL 속성으로 생성되어야 동작함
# * 영상 출력 부분의 크기만을 고려함(제목 표시줄, 창 경계는 고려되지 않음)

# 5. 영상 출력하기
# cv2.imshow(winname, mat) -> None
# winnmae : 영상을 출력할 대상 창 이름
# mat : 출력할 영상 데이터(numpy.ndarray)
# 데이터 타입에 따른 출력방식
# unit8 : 픽셀 값을 그대로 출력
# unit16, int16 : 픽셀 값을 255로 나눠서 출력
# float32, float64 : 픽셀 값에 255를 곱해서 출력
# 참고사항
# 만약 winname에 해당하는 창이 없으면 자동으로 cv2.WINDOW_ATUTOSIZE 속성의 창을 새로 만들어서 영상을 출력함
# Windows 운영체제에서는 Ctrl+C와 Ctrl+s 지원
# 실제로는 cv2.waitKey() 함수를 호출해야 화면에 영상이 나타남

# 6. 키보드 입력 대기
# cv2.wiatKey(dealy=None) -> retval
# delay : 밀리초 단위 대기 시간. delay <=0 이면 무한히 기다림, 기본값은 0
# retval : 눌린 키 값(ASCII code). 키가 눌리지 않으면 -1
# 참고사항
# cv2.waitKey() 함수는 OpenCV 창이 하나라도 있을 때 동작함
# 특정 키 입력을 확인하려면 ord() 함수를 이용
# while True :
#   if cv2.waitKey() == ord('q'):
#       break
# 주요 특수 키 코드 : ESC -> 27, ENTER -> 13, TAB -> 9

# Matplotlib을 이용한 영상 출력
# Matplotlib 패키지
# 함수 그래프, 차트(chart), 히스토그램(histogram) 등의 다양한 그리기 기능을 제공하는 Python Package
# 보통 matplotlib.pyplot 모듈을 plt 이름으로 불러와서 사용
# Jupyter Notebook에서 사용할 때 특히 유용
# pip install matplotlib
# import matplotlib.pyplot as plt

# 컬러 영상 출력
# 컬러 영상의 색상 정보가 RGB 순서여야 함
# cv2.imread() 함수로 불러온 영상의 색상 정보는 BGR순서이므로 이를 RGB 순서로 변경해야 함
# -> cv2.cvtColor() 함수 이용

# 그레이스케일 영상 출력
# plt.imshow() 함수에서 컬러맵을 cmap='gray' 으로 지정

