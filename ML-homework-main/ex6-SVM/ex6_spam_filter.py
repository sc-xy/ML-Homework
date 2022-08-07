from sklearn import svm
from scipy.io import loadmat
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

path1 = 'ex6-SVM/data/spamTrain.mat'
path2 = 'ex6-SVM/data/spamTest.mat'
train_mat, test_mat = loadmat(path1), loadmat(path2)
X, Y = train_mat['X'], train_mat['y'].ravel()
X_test, Y_test = test_mat['Xtest'], test_mat['ytest'].ravel()
svc = svm.SVC()
svc.fit(X, Y)
y_pred = svc.predict(X_test)
print(classification_report(Y_test, y_pred))

logit = LogisticRegression()
logit.fit(X, Y)
y_pred1 = logit.predict(X_test)
print(classification_report(Y_test, y_pred1))