import matplotlib.pyplot as plt 
import pandas as pd

# 와인 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# 품질 데이터별로 그룹을 나누고 수 세어보기
count_data = wine.groupby('quality')["quality"].count()
print(count_data)

# 수를 그래프로 그리기
count_data.plot()
plt.savefig("wine-count-plt.png")
plt.show()


''' 나쁜와인
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
''' # 좋은와인