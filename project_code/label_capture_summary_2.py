# import cv2
# import os,shutil
# from xml.etree.ElementTree import parse

# main_file_path = 'D:/watcher/videos/videos' # inside_croki_03,,,outsidedoor_02까지 들어있는 상위폴더
# main_save_path = 'D:/watcher/videos/newcapture' # 새로 캡쳐한 이미지들을 동영상 이름에 맞게끔 저장할 곳

# for m in os.listdir(main_file_path): # m은 inside_croki_03,,,outsidedoor_02
#     for n in os.listdir(main_file_path + '/' + m): # n은 inside_croki_03을 예로 들면 407-2, 407-3,,,411-4
#         file_path = main_file_path + '/' + m + '/' + n # file_path의 예 : D:\watcher\videos\videos\inside_croki_03\407-2
#         file_list = os.listdir(file_path) # file_list : D:\watcher\videos\videos\inside_croki_03\407-2 안의 모든 파일들(mp4와 xml)

#         print(file_list)
#         count = 0
#         for i in file_list : # 파일리스트들에서 하나씩 가져온다, i의 예 : 407-2_cam01_assault01_place04_day_spring.mp4 또는 xml
#             if i[-3:] != 'mp4': # 파일 형식이 mp4가 아닐 때 루프 건너뜀
#                 continue
#             path = file_path + '/' + i # D:\watcher\videos\videos\inside_croki_03\407-2\407-2_cam01_assault01_place04_day_spring.mp4 

#             if os.path.isdir(main_save_path + '/' + i[:-4]): # isdir : 경로가 존재하는지 확인한다
#                                                              # D:/watcher/videos/newcapture/407-2_cam01_assault01_place04_day_spring 라는 경로 존재를 확인
#                 shutil.rmtree(main_save_path + '/' + i[:-4]) # 경로(폴더)와 파일을 한꺼번에 모두 삭제(혹시나 있으면)
#             os.mkdir(main_save_path +'/' + i[:-4]) # 새로운 경로를 만든다
        

#             file_xml = i[:-3] + 'xml' # file_list 내의 xml 파일들을 file_xml이라는 변수에 대입
#             tree = parse(file_path + '/' + file_xml) # xml파일 가져오기
            
#             # xml 객체로 파싱(parsing)하기 파싱 : 문법적 해부
#             # 다른 문법의 언어의 문장을 분석하거나 문법적 관계를 해석하는 행위

#             # getroot() 함수로 xml 파일인 tree에서 root 노드를 가져온다(현재 xml 파일에서의 root node : annotation)
#             # 그냥 find()는 일치하는 첫 번째 노드를 가져온다
#             # findall()함수로 원하는 노드를 불러온다

#             # root노드 안에 object node를 찾고 object node 영역 안에 action이란 노드를 부르고 하나씩 꺼내면서 for문을 돌림
#             for action in tree.getroot().find("object").findall("action"):
#                 action_name = action.find("actionname").text # actionname에서 text를 가져와 action_name이란 변수에 대입


#                 if os.path.isdir(main_save_path + '/' + i[:-4] +'/' + action_name):
#                     shutil.rmtree(main_save_path + '/' + i[:-4]+'/' + action_name)
#                 os.mkdir(main_save_path +'/' + i[:-4]+'/' + action_name) # action_name별로 폴더를 생성
#                 print(action.findall("frame")[0].find("start").text) # action에서 frame 노드(첫번째), start 노드의 텍스트를 출력 테스트 해 본다
#                 for frame in action.findall("frame"): # action 노드 안의 frame 노드를 하나씩 불러온다
#                     vidcap = cv2.VideoCapture(path) # 동영상 파일에서 프레임을 받아온다(vidcap 변수에 대입)
#                     start = int(frame.find("start").text) - 7 # 데이터 제공 사이트에서 지정해 놓은 start 지점보다 약간씩 앞 뒤로 더 프레임을 추가해서 가져오기 위해
#                     end = int(frame.find("end").text) + 7

#                     vidcap.set(cv2.CAP_PROP_POS_FRAMES, start) 
#                     # vidcap.set() : 캡쳐하는 속성값 설정
#                     # CAP_PROP_POS_FRAMES : 현재 프레임 개수
#                     # 그 속성을 start로 지정한다? -> 이름 지정?

#                     ret = True
#                     while(ret) :
#                         ret, image = vidcap.read() # 영상을 한 프레임씩 읽는다, 제대로 읽으면 ret값이 True, 실패하면 False, image : 읽은 프레임
#                         now = int(vidcap.get(1)) 
#                         # vidcap.get(1) : 위의 img를 가리킴 
#                         # now : 읽은 이미지 숫자형으로 데이터타입 변환 
                        
#                         # 읽은 이미지가 frame에서 end 시점보다 크면 프레임 읽는 것을 멈춘다
#                         if(now > end) :
#                             break

#                         # 읽은 이미지를 7프레임당 추출    

#                         if(now % 7 == 0) : 
#                             print('Saved frame number :' + str(int(vidcap.get(1))))
#                             cv2.imwrite(main_save_path + '/' + i[:-4] + '/' + action_name +'/' + action_name + '_frame%d.jpg' % now, image) # 새롭게 .jpg 파일로 저장
#                             print('Saved frame%d.jpg' % count)
#                             count += 1

#                     vidcap.release()

# print("캡쳐 완료")