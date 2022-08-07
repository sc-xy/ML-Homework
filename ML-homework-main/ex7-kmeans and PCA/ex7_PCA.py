import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

def normalize(data):
    """对数据进行标准化处理"""
    return (data - data.mean(axis=0)) / data.std(axis=0)

def PCA(X):
    """PCA算法"""
    sigma = (X.T @ X) / X.shape[0]
    U, S, V = np.linalg.svd(sigma)
    return U, S, V

def data_dimension(X, U, k):
    """数据降维"""
    n = X.shape[1]
    if k>n:
        raise ValueError("k 应该小于 X 的维度")
    return X @ U[:,:k]

def data_recover(Z, U):
    """恢复数据"""
    n = Z.shape[1]
    if n >= U.shape[0]:
        raise ValueError("Z的维度应该小于U的维度")
    return Z @ U[:,:n].T

def data_plot(X_norm, X_recover):
    """绘制结果"""
    plt.figure(figsize=(6, 6))
    plt.scatter(X_norm[:,0], X_norm[:,1], s=20, c='b', label='original data')
    plt.scatter(X_recover[:,0], X_recover[:,1], s=20, c='r', label='recovered data')
    plt.plot([X_norm[:,0], X_recover[:,0]], [X_norm[:,1], X_recover[:,1]], '--')
    plt.legend()
    plt.show()

def data_pre(path):
    """预处理数据"""
    mat = loadmat(path)
    X = mat['X']
    X_norm = normalize(X)
    return X_norm

def main():
    """主函数"""
    path = 'ex7-kmeans and PCA/data/ex7data1.mat'
    k = 1
    X_norm = data_pre(path)
    U, S, V = PCA(X_norm)
    Z = data_dimension(X_norm, U, k)
    X_recover = data_recover(Z, U)
    data_plot(X_norm, X_recover)

if __name__ == '__main__':
    main()