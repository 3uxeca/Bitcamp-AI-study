from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42) # stratify가 뭐임

# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(X_train, y_train)
# print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
# print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

# 훈련과 테스트 정확도의 차이가 6퍼센트가 난다는건 훈련이 과적합 되어있다는 것 !
'''
훈련 세트 정확도: 0.988
테스트 세트 정확도: 0.951
'''

# 어느 column(특성)이 중요한가!
print("특성 중요도:\n", tree.feature_importances_) 

'''
특성 중요도: 30개의 column을 다 더하면 1.0이 된다!
[0.         0.         0.         0.         0.         0.
0.         0.         0.         0.         0.01019737 0.04839825
0.         0.         0.0024156  0.         0.         0.
0.         0.         0.72682851 0.0458159  0.         0.
0.0141577  0.         0.018188   0.1221132  0.01188548 0.        ]
'''