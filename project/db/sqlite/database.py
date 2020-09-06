# sqlite with Python (youtube 참고)
'''
import sqlite3
conn = sqlite3.connect('test.db') # test.db와 연결

cur = conn.cursor()
# cursor의 개념!!!
# 예를 들어 다음과 같은 student table이 있다고 가정
# id name      age
# 1. 김아무개 - 25
# 2. 이아무개 - 30
# 3. 방아무개 - 27

# 정보를 담고 있는 각 행들은 각각의 위치를 가진다 == 고유의 메모리를 가짐 == 고유의 주소를 가짐
# 각각의 위치들을 cursor라고 부를 수 있다, cursor는 현재 위치를 말한다
# select를 한다면, 이 cursor라는 것이 하나씩 움직이면서 그 정보들을 select 한다고 이해하면 된다
# 현재 가리키고 있는 것 : cursor

# 테이블 여러 개 == 데이터 베이스
# 데이터베이스가 여러 개 == 파일(?)
# 다른 프로그램들과 달리 sqllite는 데이터베이스(ex. test.db) 하나밖에 못 갖는다
# sqllite는 하나의 데이터베이스 == 하나의 파일
# test.db 라는 곳에 student, club 등의 여러 테이블이 존재함

cur.execute("select * from Student") # student table의 처음으로 이동한다 cursor가!

rows = cur.fetchall() # 시작 지점에 있는 cursor가 모든 행들을 가져 온다(엄밀히 말하면 갖고 올 준비를 하는 것, 위치에 대기만 하는 것)

# 아래가 이제 직접적으로 접근, access 하는 부분
for row in rows : #rows는 전부 갖고 있는 주소값
    print(row)

conn.close() # with를 사용하면 close 해 줄 필요가 없다
'''

'''
import sqlite3

conn = sqlite3.connect("test.db")

with conn : # conn : 아래 블록 부분을 다 실행하면 닫아주는 것이 with의 기능
    cur = conn.cursor()
    sql = "select * from Student"
    cur.execute(sql)

    rows = cur.fetchall()

    for row in rows:
        print(row)
'''

'''
"select * from student where id=? or name=?" # ?를 쓰게 되면 prepared, precompile가 됨
# 앞의 select * from student where : 이 쿼리문은 변하지 않음
# machine이 아, 뒤에 ? 쓴 부분은 니가 바꿔가면서 쓸 거구나, 하고 이해함
# 앞의 쿼리문은 메모리에 올려둔 상태가 됨

with conn :
    cur = conn.cursor()
    sql = "select * from Student where id=? or name=?"
    cur.execute(sql, (1, '김삼순')) # 첫번째 ?에는 1을, 두번째 ?에는 '김삼순'을 넣는 것 # 물음표 개수만큼 튜플 개수도 결정
    rows = cur.fetchall()

    for row in rows :
        print(row)

'''