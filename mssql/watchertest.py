# import pymssql as ms
# print("접속 완료")

# conn = ms.connect(server = '127.0.0.1', user = 'bit2', password = '1234', database = 'bitdb')

# with conn :
#     cur = conn.cursor()
#     sql = "SELECT * FROM watchertest;"
#     cur.execute(sql)

#     rows = cur.fetchall()

#     for row in rows :
#         print(row)

# '''
# (1, 'test61113spring', 'mp4', 'C:\\dbtest')
# (2, 'test61113summer', 'mp4', 'C:\\dbtest')
# (3, 'test61113winter', 'mp4', 'C:\\dbtest')
# '''

import pymssql as ms
print("접속 완료")

conn = ms.connect(server = '127.0.0.1', user = 'bit2', password = '1234', database = 'teamproject')

with conn :
    cur = conn.cursor()
    sql = "SELECT * FROM videolist;"
    cur.execute(sql)

    rows = cur.fetchall()

    for row in rows :
        print(row)