import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import sys

def load_data(path):
    """数据处理"""
    mat = loadmat(path)
    data = pd.DataFrame(mat['X'], columns=['x1', 'x2'])
    data['Y'] = mat['y']
    return data

def decision_boundary(svc, x1min, x1max, x2min, x2max):
    """决策边界"""
    x1 = np.arange(x1min, x1max, 0.01)
    x2 = np.arange(x2min, x2max, 0.01)
    l = [(x, y) for x in x1 for y in x2]
    x1, x2 = zip(*l)
    c = pd.DataFrame({'x1':x1, 'x2':x2})
    c['val'] = svc.decision_function(c[['x1', 'x2']])
    c = c[np.abs(c['val'])<= 2*10**-3]    
    return c

def plot_data(positive, negative, c, C):
    """数据可视化"""
    plt.figure(figsize=(6, 4))
    plt.scatter(positive['x1'], positive['x2'], marker='+', c='b', s=50, label='positive')
    plt.scatter(negative['x1'], negative['x2'], marker='o', c='y', s=50, label='negative')
    plt.plot(c['x1'], c['x2'], label=f'C={C}')
    plt.legend(loc='best')
    plt.show()

def main():
    """主函数"""
    if (len(sys.argv))==2:
        try:
            C = float(sys.argv[1])
        except ValueError:
            print("请输入正确的参数：  数字a --参数C")
            sys.exit()
    elif (len(sys.argv))==1:
        C = 1
    else:
        print("请输入正确的参数：  数字a --参数C")
        sys.exit()
    path = 'ex6-SVM/data/ex6data1.mat'
    data = load_data(path)
    positive = data[data['Y'] == 1]
    negative = data[data['Y'] == 0]
    svc = svm.LinearSVC(C=C, loss='hinge', max_iter=20000)
    svc.fit(data[['x1', 'x2']], data['Y'])
    c = decision_boundary(svc, data['x1'].min(), data['x1'].max(), data['x2'].min(), data['x2'].max())
    plot_data(positive, negative, c, C)

if __name__ == '__main__':
    main()