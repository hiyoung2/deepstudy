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

# cv2.VideoCapture Class
# Opencv에서는 Camera와 Video로부터 Frame을 받아오는 작업을 cv2.VideoCapture Class 하나로 처리
# 하나 짚고 넘어갈 점!
# 영상이라는 용어를 많이 사용하게 될 텐데, 보통 사람들 사이에서의 영상은 흔히 아는 video를 가리키는데
# 이미지 프로세싱 전공쪽에서는 영상은 정지영상, 한 장의 사진을 영상으로 하고
# 우리가 흔히 사용하는 의미의 영상은 예를 들면 유튜브 영상 같은 것은 '동영상'이라고 구분지어 표현하는 경우가 많다
# 즉!
# 영상처리라 함은, 동영상을 처리한다는 의미가 아니라 한 장 한 장, 사진 파일들을 처리하는 것으로 이해하면 된다
# 일반적 용어 사용과 Gap이 있으니 이해를 해야 한다

# 카메라나 동영상을 처리하기 위해서는
# 결과적으로는 한 장의 정지영상을 가져와서 한 장씩 화면에 보여주고, 처리하고 보여주고 이것이 반복되는 것이다
# 영화를 예로 들면 초당 24 또는 30프레임을 계속 보여주면 잔상효과로 움직이는 것처럼 보여지듯이
# 실제로 카메라로 들어온 영상에서 한 프레임은 정지영상이다
# 카메라, 동영상에서 정지 영상을 한 장씩 받아 올 수 있는 인터페이스만 있다면
# 동영상이든 카메라든 움직이듯이 보여줄 수가 있다

# 7. 동영상, 정지 영상 시퀀스, 비디오 스트림 열기
# 1) cv2.VideoCapture(filename, apiPreference=None) -> retval
# filename : 
# (1) 비디오 파일 이름 (e.g. 'video.avi')
# (2) 정지 영상 시퀀스 (e.g. 'img_%02d.jpg')
# (3) 비디오 스트림 URL (e.g. 'protocol://host:port/script?params|auth')
# apiPreference : 선호하는 동영상 처리 방법을 지정
# retval : cv2.VideoCapture 객체

# 2) cv2.VideoCaptrue.open(filename, apiPreference=None) -> retval
# retval : 성공하면 True, 실패하면 False

# 8. 카메라 열기
# 1) cv2.VideoCapture(index, apiPreference=None) -> retval
# index : camera_id + domain_offset(CAP_*)id 
# (1) camera_id==0 이면 시스템 기본 카메라
# (2) domain_offset==0 이면 auto detect
# (3) 기본 카메라를 기본 방법으로 열려면 index에 0을 전달
# apiPreference : 선호하는 카메라 처리 방법을 지정
# retval : cv2.VideoCapture 객체

# 2) cv2.VideoCaptrue.open(index, apiPreference=None) -> retval
# retval : 성공하면 True, 실패하면 False

# 9. 비디오 캡쳐가 준비되었는지 확인
# cv2.VideoCapture.isOpened() -> retval
# retval : 성공하면 True, 실패하면 False

# 10. Frame 받아오기
# cv2.VideoCapture.read(image=None) -> retval, image
# retval : 성공 True, 실패 False
# image : 현재 프레임(numpy.ndarray)

# 11. 카메라, 비디오 장치 속성 값 참조
# cv2.VideoCaptrue.get(propId) -> retval
# propld : 속성 상수
# cv2.CAP_PROP_FRAME_WIDTH : 프레임 가로 크기
# cv2.CAP_PROP_FRAME_HEIGHT : 프레임 세로 크기
# cv2.CAP_PROP_FPS : 초당 프레임 수
# cv2.CAP_PROP_POS_MSEC : 밀리초 단위로 현재 위치
# cv2.CAP_PROP_POS_FRAMES : 현재 프레임 번호
# cv2.CAP_PROP_EXPOSURE : 노출
# cv2.CAP_PROP_ZOOM : 줌(확대/축소 비율)
# retval : 성공하면 해당 속성 값, 실패하면 0

# 12. 카메라, 비디오 장치 속성 값 참조
# cv2.VideoCapture.set(propId, value) -> retval
# propId : 속성 상수
# value : 속성 값
# retval : 성공하면 True, 실패하면 False

# 13. 동영상 파일 저장하기
# 1) cv2.VideoWriter Class
# OpenCV에서는 cv2.VideoWriter 클래스를 이용하여 일련의 프레임을 동영상 파일로 저장 가능
# 일련의 프레임은 모두 크기와 데이터 타입이 같아야 한다

# 2) Fourcc(4-문자코드, four character code)
# 동영상 파일의 코덱, 압축 방식, 색상 혹은 픽셀 포맷 등을 정의하는 정수 값
# http://www.fourcc.org/codecs/php
# cv2.VideoWriter_fourcc(*'DIVX') : DIVX MPEG-4 코덱
# cv2.VideoWriter_fourcc(*'XVID') : XVID MPEG-4 코덱
# cv2.VideoWriter_fourcc(*'FMP4') : FMPEG MPEG-4 코덱
# cv2.VideoWriter_fourcc(*'X264') : H.264/AVC 코덱
# cv2.VideoWriter_fourcc(*'MJPG') : Motion-JPEG 코덱

# 14. 저장을 위한 동영상 파일 열기
# cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor=None) -> retval
# filename : 비디오 파일 이름(e.g. 'video.mp4')
# fourcc : fourcc(e.g. cv2.VideoWriter_fourcc(*'DIVX))
# fps : 초당 프레임 수 (e.g. 30)
# frameSize : 프레임 크기(e.g. [640, 480])
# isColor : if is Color == True , else == False
# retval: cv2.VideoWriter 객체

# cv2.VideoWriter.open(filename, fourcc, fps, frameSize, isColor=None) -> reval
# retval : 성공하면 True, 실패하면 False

# 15. 비디오 파일이 준비되었는지 확인
# cv2.VideoWriter.isOpend() -> retval
# retval : 성공 True, 실패 False

# 16. 프레임 받아오기
# cv2.VideoWriter.write(image) -> None
# image : 저장할 프레임(numpy.ndarray)

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


