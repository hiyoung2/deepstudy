import glob 
from sklearn.model_selection import train_test_split
def file_path_save(): 
    trainlist = []
    vallist = []

    files = sorted(glob.glob("C:/darknet-master/darknet-master/build/darknet/x64/mydata/testdata/*.jpg")) 
    train_files, val_files = train_test_split(
        files, shuffle=True, train_size=0.8, random_state=66
    )
    print(len(train_files))
    print(len(val_files))

    trainlist.append(train_files)
    vallist.append(val_files)

    print(trainlist)
    print("========================================================")
    print(vallist)

    # for i in range(len(train_files)): 
    #     f = open("C:/darknet-master/darknet-master/build/darknet/x64/mydata/train.txt", 'a') 
    #     f.write(train_files[i] + "\n") 

    for i in range(len(val_files)): 
        f = open("C:/darknet-master/darknet-master/build/darknet/x64/mydata/val.txt", 'a') 
        f.write(val_files[i] + "\n") 

if __name__ == '__main__': file_path_save()

# ~~ testdata/*.jpg : *로 인해 jpg 형식 파일 모두를 불러낼 수 있다
# SELECT * FROM 이 생각난다

# glob.glob 함수를 통해 특정 디렉토리(폴더)로 부터 디렉토리 및 파일 목록을 가져와서 리스트를 보면 정렬되지 않는 경우
# sorted : 파일명 순서대로 정렬된다
# if, 다른 조건으로 정렬하려면?

# sorted(glob.glob('*'), key = os.path.getctime) # 파일 생성일
# sorted(glob.glob('*'), key = os.path.getatime) # 파일 최근 접근일
# sorted(glob.glob('*'), key = os.path.getmtime) # 파일 최종 수정일
# sorted(glob.glob('*'), key = os.path.getsize) # 파일 사이즈로 정렬


# open
# r : read(default), 읽기용으로 파일 열기
# w : write, 쓰기용으로 파일 열기
# - 파일이 존재하지 않으면 새로 생성, 파일이 존재하면 파일 내용을 비움(turncates the file)
# x : create, 새로운 파일 생성, 만약 파일이 이미 존재하면 IOError 예외 발생
# a : append

# r : 읽기모드 - 파일을 읽기만 할 때 사용
# w : 쓰기모드 - 파일에 내용을 쓸 때 사용
# a : 추가모드 - 파일의 마지막에 새로운 내용을 추가 시킬 때 사용

# 파일을 쓰기모드로 열면 해당 파일이 존재할 경우 원래 있던 내용이 모두 사라지고,
# 해당 파일이 존재하지 않으면 새로운 파일이 생성된다
# 원래 있던 값을 유지하면서 단지 새로운 값만 추가해야하는 경우에 a 모드를 사용한다
#