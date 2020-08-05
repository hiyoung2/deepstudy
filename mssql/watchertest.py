import pymssql as ms
print("접속 완료")

conn = ms.connect(server = '127.0.0.1', user = 'bit2', password = '1234', database = 'bitdb')

with conn :
    cur = conn.cursor()
    sql = "SELECT * FROM watchertest;"
    cur.execute(sql)

    rows = cur.fetchall()

    for row in rows :
        print(row)

