import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

#붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding = "utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# classifier 알고리즘 모두 추출하기
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

# K-분할 크로스 밸리데이션 전용 객체
kfold_cv = KFold(n_splits=3, shuffle=True) 
            # 5개로 나누면 4개는 train, 1개는 test. 31개의 분류기로 돌리면 5번씩 총 155번 돌아가는 것

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기
    clf = algorithm()

    # score 메서드를 가진 클래스를 대상으로 하기
    if hasattr(clf,"score"): # hasattr는 attribute가 있는지 여부를 확인할 때 사용

        # 크로스 밸리데이션
        scores = cross_val_score(clf, x, y, cv=kfold_cv) # 돌린 score가 어떻게 되냐
        print(name, "의 정답률=")
        print(scores, "평균:", scores.mean(), "합:", scores.sum())

'''
 # n_splits = 3
1. LinearDiscriminantAnalysis / mean=0.98, sum=2.94
2. MLPClassifier / mean=0.9733, sum=2.92
3. QuadraticDiscriminantAnalysis, mean=0.9666, sum=2.9
   RadiusNeighborsClassifier, mean=0.9666, sum=2.9
   SVC, mean=0.9666, sum=2.9

# n_splits = 4
1. KNeighborsClassifier / mean=0.9802, sum=3.9203
2. LinearDiscriminantAnalysis / mean=0.98, sum=3.9210
3. SVC / mean=0.9735, sum=3.8940

# n_splits = 5
1. LinearDiscriminantAnalysis / mean=0.98, sum=4.9
2. KNeighborsClassifier / mean=0.9733, sum=4.8666
3. LabelPropagation, / mean=0.9666, sum=4.8333
   LabelSpreading, / mean=0.9666, sum=4.8333
   MLPClassifier, / mean=0.9666, sum=4.8333
   NuSVC, / mean=0.9666, sum=4.8333
   QuadraticDiscriminantAnalysis / mean=0.9666, sum=4.8333

# n_splits = 6
1. LinearDiscriminantAnalysis / mean=0.98, sum=5.88
2. LabelSpreading, / mean=0.9666, sum=5.8
   LinearDiscriminantAnalysis, / mean=0.9666, sum=5.8
   LinearSVC, / mean=0.9666, sum=5.8
   QuadraticDiscriminantAnalysis, / mean=0.9666, sum=5.8
   SVC / mean=0.9666, sum=5.8
3. BaggingClassifier / mean=0.96, sum=5.76

# n_splits = 7 
1. MLPClassifier / mean=0.9802, sum=6.8614
2. SVC / mean=0.9799, sum=6.8593
3. LinearDiscriminantAnalysis / mean=0.9737, sum=6.8160

# n_splits = 8 
1. LinearDiscriminantAnalysis / mean=0.9802, sum=7.8421
2. SVC / mean=0.9733, sum=7.7865
3. MLPClassifier / mean=0.9729, sum=7.7836

# n_splits = 9
1. LinearDiscriminantAnalysis / mean=0.9803, sum=8.8235
2. SVC / mean=0.9738, sum=8.7647
3. MLPClassifier / mean=0.9669, sum=8.7022

# n_splits = 10
1. LinearDiscriminantAnalysis / mean=0.98, sum=9.8
2. QuadraticDiscriminantAnalysis, / mean=0.9733, sum=9.73
   SVC / mean=0.9733, sum=9.73
3. MLPClassifier, / mean=0.9666, sum=9.6666
   LabelSpreading, / mean=0.9666, sum=9.6666
   LabelPropagation, / mean=0.9666, sum=9.6666
   KNeighborsClassifier, / mean=0.9666, sum=9.6666
   BaggingClassifier / mean=0.9666, sum=9.6666
'''       