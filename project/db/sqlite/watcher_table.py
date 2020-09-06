import sqlite3

conn = sqlite3.connect("test.db")

print("연결 완료")

cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS watcher(ID INTEGER, Title TEXT, FF TEXT, Dir TEXT)""")

sql = "DELETE FROM watcher"
cursor.execute(sql)

sql = "INSERT into watcher(ID, Title, FF, Dir) values (?, ?, ?, ?)"
cursor.execute(sql, (1, 'test61113spring', 'mp4', 'C:\dbtest'))

sql = "INSERT into watcher(ID, Title, FF, Dir) values (?, ?, ?, ?)"
cursor.execute(sql, (2, 'test61113summer', 'mp4', 'C:\dbtest'))

sql = "INSERT into watcher(ID, Title, FF, Dir) values (?, ?, ?, ?)"
cursor.execute(sql, (3, 'test61113winter', 'mp4', 'C:\dbtest'))


# 데이터 조회
sql = "SELECT * FROM watcher"
cursor.execute(sql)

rows = cursor.fetchall()


for row in rows :
    print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " + str(row[3]) + " ")



conn.commit() # github 커밋처럼 commit을 해줘야 저장이 된다
conn.close()