import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import sys

def load_data(path):
    """处理数据"""
    mat = loadmat(path)
    training = pd.DataFrame(mat['X'], columns=['x1', 'x2'])
    training['Y'] = mat['y']
    cv = pd.DataFrame(mat['Xval'], columns=['x1', 'x2'])
    cv['Y'] = mat['yval']
    return training, cv

def find_best(training, cv):
    """寻找最合适的参数"""
    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    search = []
    combination = [(C, gamma) for C in candidate for gamma in candidate]
    for C, gamma in combination:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(training[['x1', 'x2']], training['Y'])
        search.append(svc.score(cv[['x1', 'x2']], cv['Y']))
    return combination[np.argmax(search)][0], combination[np.argmax(search)][1]

def get_boundary(training, C, gamma):
    """决策边界"""
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(training[['x1', 'x2']], training['Y'])
    x1 = np.linspace(training['x1'].min(), training['x1'].max(), 1000)
    x2 = np.linspace(training['x2'].min(), training['x2'].max(), 1000)
    l = [(x, y) for x in x1 for y in x2]
    x1, x2 = zip(*l)
    c = pd.DataFrame({'x1':x1, 'x2':x2})
    c['val'] = svc.decision_function(c[['x1', 'x2']])
    c = c[np.abs(c['val']) <= 2*10**-3]
    return c

def plot_data(positive, negative, c):
    """绘制数据""" 
    plt.figure(figsize=(6, 4))
    plt.scatter(positive['x1'], positive['x2'], marker='+', s=50, c='r', label='positive')
    plt.scatter(negative['x1'], negative['x2'], marker='o', s=50, c='b', label='negative')
    plt.scatter(c['x1'], c['x2'], s=10, label='decision', c='g')
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def main():
    """主函数"""
    path = 'ex6-SVM/data/ex6data3.mat'
    training, cv = load_data(path)
    positive = training[training['Y']==1]
    negative = training[training['Y']==0] 
    C, gamma = find_best(training, cv)
    c = get_boundary(training, C, gamma)
    plot_data(positive, negative, c)

if __name__ == '__main__':
    main()