from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # decisiontree를 ensemble했기 때문에 ensemble에 있다!
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42) # stratify가 뭐임

# tree = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0, criterion='gini', max_depth=4, min_samples_split=50, min_samples_leaf=5)
# tree.fit(X_train, y_train)
# print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
# print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

tree = RandomForestClassifier(max_depth=150)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

# tree = RandomForestClassifier(max_depth=4, random_state=0)
# tree.fit(X_train, y_train)
# print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
# print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

# 훈련과 테스트 정확도의 차이가 6퍼센트가 난다는건 훈련이 과적합 되어있다는 것 !

# RF 꿀팁
# n_estimators : 클수록 좋다. 단점 : memory를 많이 차지한다. 기본값 : 100
# n_jobs = -1 : cpu 병렬처리. 작업관리자 들어가서 성능 -> cpu의 코어 수가 n_jobs에 입력할 수 있는 최대값
# max_features : 기본값 써라!

# 어느 column(특성)이 중요한가!
print("특성 중요도:\n", tree.feature_importances_) 

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)
plt.show()