import mysql.connector

config = {
    'user':'root', 
    'password':'youngbora',
    'host':'127.0.0.1',
    'database':'pydb',
    'port':'3306'
}

# connection 생성
def getConn():
    conn = mysql.connector.connect(**config) # ** : 가변인자 , 딕셔너리형태 config를 받는다
    return conn

# main에서 사용해보자 -> p01.py 파일로

# *config : 튜플형식
# **conofig : 딕셔너리 형태


if __name__ == '__main__': getConn()

print("접속 완료")