import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys

def data_pre(path):
    """预处理数据"""
    imc = io.imread(path) / 255
    data = imc.copy().reshape(128*128, 3)
    centers = random_init(data, 16)
    return imc, data, centers

def random_init(X, k):
    """随机选取样本初始化"""
    idx = np.random.randint(0, len(X), k)
    return X[idx]

def find_centers(X, centers):
    """寻找新的聚类点"""
    idx = np.zeros(len(X))
    for i in range(len(X)):
        mindist = np.sum((X[i,:]-centers[0,:])**2)
        for j in range(1, len(centers)):
            dist = np.sum((X[i,:]-centers[j,:])**2)
            if dist<mindist:
                idx[i] = j
                mindist = dist
    return idx

def compute_centers(X, idx, k):
    """计算新的聚类点"""
    centers = []
    for i in range(k):
        centers_i = np.mean(X[idx==i], axis=0)
        centers.append(centers_i)
    return np.array(centers)

def run_Kmeans(X, centers, iters, k=16):
    """执行Kmeans算法"""
    for i in range(iters):
        idx = find_centers(X, centers)
        centers = compute_centers(X, idx, k)
    return idx, centers

def plot_compare(imc, data):
    """画出对比图"""
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(imc)  
    imc = data.reshape(128, 128, 3)
    ax[1].imshow(imc)
    plt.show()

def main():
    """主函数"""
    path = 'ex7-kmeans and PCA/data/bird_small.png'
    imc = io.imread(path) / 255
    imc, data, centers = data_pre(path)
    k, iters = 16, 10
    if len(sys.argv)==2:
        if int(sys.argv[1])==1:
            """手搓Kmeans"""
            idx, centers = run_Kmeans(data, centers, iters, k)
            for i in range(k):
                data[idx==i] = centers[i]
        elif int(sys.argv[1])==2:
            """库函数"""
            model = KMeans(n_clusters=k, n_init=iters)
            model.fit(data)
            centers = model.cluster_centers_
            C = model.predict(data)
            data = centers[C].reshape(128, 128, 3)
        else:
            print("请输入正确的参数：\n\t1.\t手搓Kmeans\t\t2.\t库函数")
            sys.exit()
    elif len(sys.argv)==1:
        """手搓"""
        idx, centers = run_Kmeans(data, centers, iters, k)
        for i in range(k):
            data[idx==i] = centers[i]
    else:
        print("请输入正确的参数：\n\t1.\t手搓Kmeans\t\t2.\t库函数")
        sys.exit()
    plot_compare(imc, data)

if __name__ == '__main__':
    main()