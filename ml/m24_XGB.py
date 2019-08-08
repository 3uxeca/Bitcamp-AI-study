# XG Boost

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42) # stratify가 뭐임

# tree = XGBClassifier(random_state=0)
# tree.fit(X_train, y_train)
# print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
# print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

tree = XGBClassifier(n_estimator=10, learning_rate=1.5, max_depth=1, eval_metric='error',
                    base_score=0.4, early_stopping_rounds=10, n_jobs=-1)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

# 훈련과 테스트 정확도의 차이가 6퍼센트가 난다는건 훈련이 과적합 되어있다는 것 !

# 어느 column(특성)이 중요한가!
print("특성 중요도:\n", tree.feature_importances_) 

# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), cancer.feature_names)
#     plt.xlabel("특성 중요도")
#     plt.ylabel("특성")
#     plt.ylim(-1, n_features)

# plot_feature_importances_cancer(tree)
# plt.show()