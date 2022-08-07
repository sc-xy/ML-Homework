import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA as sk_PCA
import sys

def plot_n_imgs(X, n):
    """绘制n张图像，n必须是个平方数"""
    pic_sz = int(np.sqrt(X.shape[1]))
    gid_sz = int(np.sqrt(n))
    img = X[:n,:]
    fig, ax = plt.subplots(nrows=gid_sz, ncols=gid_sz, figsize=(6, 4), sharex=True, sharey=True)

    for i in range(gid_sz):
        for j in range(gid_sz):
            ax[i, j].imshow(img[i*gid_sz+j].reshape(pic_sz, pic_sz).T)
            ax[i, j].set_xticks(np.array([]))
            ax[i, j].set_yticks(np.array([]))
    plt.show()
    
def PCA(X):
    """运行PCA算法"""
    sigma = (X.T @ X) / X.shape[0]
    U, S, V = np.linalg.svd(sigma)
    return U, S, V

def reduce_data(X, U, k):
    """对数据进行降维"""
    n = X.shape[1]
    if k > n:
        raise ValueError("k 应该小于 X 的维度")
    return X @ U[:,:k]

def recover_data(Z, U):
    """预测数据"""
    n = Z.shape[1]
    if n>=U.shape[1]:
        raise ValueError("Z的维度应该小于U的维度")
    return Z @ U[:,:n].T

def data_pre(path):
    """处理数据"""
    mat = loadmat(path)
    X = mat["X"]
    return X

def main():
    """主函数"""
    path = 'ex7-kmeans and PCA/data/ex7faces.mat'
    X = data_pre(path)
    k, n = 100, 100
    if len(sys.argv)==2:
        if int(sys.argv[1])==1:
            """手搓Kmeans"""
            U, S, V = PCA(X)
            Z = reduce_data(X, U, k=100)
            X_recover = recover_data(Z, U)
        elif int(sys.argv[1])==2:
            """库函数"""
            pca = sk_PCA(n_components=100)
            Z = pca.fit_transform(X)
            X_recover = pca.inverse_transform(Z)
        else:
            print("请输入正确的参数：\n\t1.\t手搓PCA(默认)\t\t2.\t库函数")
            sys.exit()
    elif len(sys.argv)==1:
        """手搓"""
        print("请输入正确的参数：\n\t1.\t手搓PCA(默认)\t\t2.\t库函数")
        U, S, V = PCA(X)
        Z = reduce_data(X, U, k=100)
        X_recover = recover_data(Z, U)
    else:
        print("请输入正确的参数：\n\t1.\t手搓PCA(默认)\t\t2.\t库函数")
        sys.exit()
    plot_n_imgs(X, 100)
    plot_n_imgs(X_recover, 100)

if __name__ == '__main__':
    main()
