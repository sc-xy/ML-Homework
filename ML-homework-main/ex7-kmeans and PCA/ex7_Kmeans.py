import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans
import sys

def random_init(data, k):
    """随机选择K个样本做初始质心"""
    idx = np.random.randint(0, len(data), k)
    return data[idx]

def combine_with_C(data, C):
    """数据连接上C"""
    data_with_C = data.copy()
    data_with_C['C'] = C
    return data_with_C

def find_centers(X, centers):
    """寻找最近的聚类点"""
    idx = np.zeros(len(X))
    for i in range(len(X)):
        min_dist = np.sum((X[i, :]-centers[0,:])**2)
        for j in range(1, len(centers)):
            dist = np.sum((X[i,:]-centers[j,:])**2)
            if dist<min_dist:
                idx[i] = j
                min_dist = dist
    return idx

def run_Kmeans(X, centers, iters, k=3):
    """执行K_means算法"""
    for i in range(iters):
        idx = find_centers(X, centers)
        centers = compute_centers(X, idx, k)
    return idx, centers

def compute_centers(X, idx, k):
    """计算新的聚类点"""    
    centers = []
    for i in range(k):
        centers_i = np.mean(X[idx==i], axis=0)
        centers.append(centers_i)
    return np.array(centers)

def plot_clusters(X, idx):
    """画出聚类结果"""
    cluster0 = X[idx==0]
    cluster1 = X[idx==1]
    cluster2 = X[idx==2]
    plt.figure(figsize=(6, 4))
    plt.scatter(cluster0[:, 0], cluster0[:, 1], c='r', s=20, label='cluster1')
    plt.scatter(cluster1[:, 0], cluster1[:, 1], c='b', s=20, label='cluster2')
    plt.scatter(cluster2[:, 0], cluster2[:, 1], c='g', s=20, label='cluster3')
    plt.legend()
    plt.show()

def main():
    """主函数"""    
    path = 'ex7-kmeans and PCA/data/ex7data2.mat'
    data = loadmat(path)
    X = pd.DataFrame(data['X'], columns=[['x1', 'x2']]).values
    k = 3
    if len(sys.argv)==2:
        if int(sys.argv[1])==1:
            """手搓Kmeans"""
            centers = random_init(X, k)
            idx, centers = run_Kmeans(X, centers, 10, k)
        elif int(sys.argv[1])==2:
            """库函数"""
            sk_kmeans = KMeans(n_clusters=k)
            sk_kmeans.fit(X)
            idx = sk_kmeans.predict(X)
        else:
            print("请输入正确的参数：\n\t1.\t手搓Kmeans\t\t2.\t库函数")
            sys.exit()
    elif len(sys.argv)==1:
        """手搓"""
        centers = random_init(X, k)
        idx, centers = run_Kmeans(X, centers, 10, k)
    else:
        print("请输入正确的参数：\n\t1.\t手搓Kmeans\t\t2.\t库函数")
        sys.exit()
    plot_clusters(X, idx)

if __name__ == '__main__':
    main()