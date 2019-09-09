# 모듈 불러오기
import pymssql as ms
import numpy as np

# 데이터베이스에 연결
conn = ms.connect(server='127.0.0.1', user='bit', password='1234', database='bitdb')

# 커서 만들기
cursor = conn.cursor()

# 커서에 쿼리를 입력해 실행
cursor.execute('SELECT * FROM iris2;')

row = cursor.fetchall()
# print(row)
# print(type(row))
conn.close()

aaa = np.asarray(row)
print(aaa)
print(type(aaa), aaa.shape) # (150, 5)

np.save('test_aaa.npy',aaa)

'''
# 한 행 가져오기
row = cursor.fetchone()
# print(type(row))      # tuple

while row:
    print("첫컬럼=%s, 둘컬럼=%s" %(row[0], row[1]))
    # print(row)
    row = cursor.fetchone()

# 연결 닫기
conn.close()
'''