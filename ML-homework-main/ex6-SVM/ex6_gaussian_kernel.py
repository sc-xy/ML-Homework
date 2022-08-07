import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm 
import sys

def gaussian_kernel(x1, x2, sigma):
    """高斯核函数"""
    return np.exp(- np.power((x1 - x2), 2).sum() / (2*sigma**2))

def load_data(path):
    """处理数据""" 
    mat = loadmat(path)
    data = pd.DataFrame(mat['X'], columns=['x1', 'x2'])
    data['Y'] = mat['y']
    return data, data[data['Y']==1], data[data['Y']==0]

def plot_data(positive, negative, c, gamma):
    """绘制数据"""
    plt.figure(figsize=(6, 4))
    plt.scatter(positive['x1'], positive['x2'], marker='o', s=50, c='b', label='positive')
    plt.scatter(negative['x1'], negative['x2'], marker='x', s=50, c='r', label='negative')
    plt.scatter(c['x1'], c['x2'], label=f'C=100,g={gamma}', s=10, c='g')
    plt.legend(loc='best')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def decision_boundary(data, gamma):
    """决策边界"""
    svc = svm.SVC(C=100, kernel='rbf', gamma=gamma)
    svc.fit(data[['x1', 'x2']], data['Y'])
    x1 = np.linspace(min(data['x1']), max(data['x1']), 1000)
    x2 = np.linspace(min(data['x2']), max(data['x2']), 1000)
    l = [(x, y) for x in x1 for y in x2]
    x1, x2 = zip(*l)
    c = pd.DataFrame({'x1':x1, 'x2':x2})
    c['val'] = svc.decision_function(c[['x1', 'x2']])
    c = c[np.abs(c['val']) <= 2*10**-3]
    return c

def main():
    """主函数"""
    if len(sys.argv)==1:
        gamma = 10
    elif len(sys.argv)==2:
        try:
            gamma = float(sys.argv[1])

        except ValueError:
            print('请输入正确的参数: gamma(默认10)')
            sys.exit()
    else:
        print('请输入正确的参数: gamma(默认10)')
        sys.exit()
    path = 'ex6-SVM/data/ex6data2.mat'
    data, positive, negative = load_data(path)
    c = decision_boundary(data, gamma)
    plot_data(positive, negative, c, gamma)

if __name__=='__main__':
    main()
