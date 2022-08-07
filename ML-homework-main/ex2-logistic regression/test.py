import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

t1 = np.linspace(-1.5, 1.5, 1000)
t2 = np.linspace(-1.5, 1.5, 1000)
cordinates = [(x, y) for x in t1 for y in t2]

x_cord, y_cord = zip(*cordinates)
mapped = pd.DataFrame({'f10': x_cord, 'f01': y_cord,'f20': np.power(x_cord, 2), 'f02': np.power(y_cord, 2)})
mapped.insert(0, 'ones', 1)

pred = mapped.values @ np.array([-1, 0, 0, 1, 1]).T
decision = mapped[ np.abs(pred) <= 2*10**-3]
x = decision.f10
y = decision.f01
plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=10, marker='.')
plt.show()