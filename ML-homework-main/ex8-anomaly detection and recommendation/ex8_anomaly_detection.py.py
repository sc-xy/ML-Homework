import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import sys

def gaussian(X):
    """实现高斯分布，返回均值和方差"""
    mu = X.mean(axis=0)
    sigma = ((X - mu)**2).sum(axis=0) / X.shape[0]
    return mu, sigma

def pdf(X, mu, sigma):
    """求概率"""
    n = X.shape[1]
    ret = 1 / (np.power(2*np.pi, n/2) * sigma) * np.exp(- (X - mu)**2 / (2*sigma))
    ret = ret.prod(axis=2)
    return ret

def threshold_selection(X, Xval, yval):
    """选择合适的阈值"""
    mu = X.mean(axis=0)
    cov = np.cov(X.T)

    # 通过X得出Xval的预测
    multi_normal = stats.multivariate_normal(mu, cov)
    pval = multi_normal.pdf(Xval)
    
    epslion = np.linspace(np.min(pval), np.max(pval), num=10000)
    fs = []
    for i in epslion:
        y_pred = (pval <= i).astype('int')
        fs.append(f1_score(yval, y_pred))

    max_fs = np.argmax(fs)
    return epslion[max_fs], fs[max_fs]

def predict(X, Xval, e, Xtest, ytest):
    """预测函数"""
    Xdata = np.concatenate((X, Xval), axis=0)
    mu = Xdata.mean(axis=0)
    cov = np.cov(Xdata.T)
    multi_normal = stats.multivariate_normal(mu, cov)
    pval = multi_normal.pdf(Xtest)
    y_pred = (pval <= e).astype('int')
    print(classification_report(ytest, y_pred))
    return multi_normal, y_pred

def data_pre(path):
    """处理数据"""
    mat = loadmat(path)
    X = mat['X']
    Xval, Xtest, yval, ytest = train_test_split(mat.get('Xval'), mat.get('yval'), test_size=0.5)
    return X, Xval, Xtest, yval, ytest

def plot_data(x, y, pos, multi_normal, Xtest, y_pred):
    """绘制结果"""
    plt.figure(figsize=(8, 8))
    plt.contour(x, y, multi_normal.pdf(pos), colors='black')
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughtput (Mb/s)")
    plt.scatter(Xtest[:,0], Xtest[:,1], marker='o', c='b', s=20)
    plt.scatter(Xtest[y_pred==1][:,0], Xtest[y_pred==1][:,1], marker='x', c='r', s=50)
    plt.show()

def main():
    """主函数"""
    path1 = 'ex8-anomaly detection and recommendation/data/ex8data1.mat'
    path2 = 'ex8-anomaly detection and recommendation/data/ex8data2.mat'
    if len(sys.argv)==2:
        if int(sys.argv[1])==1:
            path = path1
        elif int(sys.argv[1])==2:
            path = path2
        else:
            print("请输入正确的参数：\n\t1.\t低维误差分析\t\t2.\t高维误差分析")
            sys.exit()
    elif len(sys.argv)==1:
        path = path1
        print("请输入正确的参数：\n\t1.\t低维误差分析\t\t2.\t高维误差分析")
    else:
        print("请输入正确的参数：\n\t1.\t低维误差分析\t\t2.\t高维误差分析")
        sys.exit()
    X, Xval, Xtest, yval, ytest = data_pre(path)
    x, y = np.linspace(0, 25, 100), np.linspace(0, 25, 100)
    x, y = np.meshgrid(x, y)
    pos = np.dstack((x, y))
    e, fs = threshold_selection(X, Xval, yval)
    print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))
    multi_normal, y_pred = predict(X, Xval, e, Xtest, ytest)
    if len(sys.argv)==2 and int(sys.argv[1])==2:
        print('find {} anamolies in Xtest data set'.format(y_pred.sum()))
        yval = multi_normal.pdf(X)
        y_pred = (yval <= e).astype('int')
        print('find {} anamolies in X data set'.format(y_pred.sum()))
    else:
        plot_data(x, y, pos, multi_normal, Xtest, y_pred)

if __name__ == '__main__':
    main()
