import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y=cancer.target
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)

# print(x_train,y_train)

classes=["malignant",'benign']
# classes=['setosa' 'versicolor' 'virginica']
kernals=["linear","poly","sigmoid","rbf"]
for i in kernals:
    # clf = KNeighborsClassifier(n_neighbors=i)
    clf=svm.SVC(kernel=i)
    clf.fit(x_train,y_train)

    cls_predict=clf.predict(x_test)
    # print(cls_predict)
    acc=metrics.accuracy_score(y_test,cls_predict)
    print(f'accuracy on using kernal={i}',acc)

# plt.plot(cls_predict,y_test)
# plt.show()