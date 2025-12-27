import numpy as np

x = np.array([1,2,3,4])
y = np.array([2,4,6,8])

m = ((x-x.mean())*(y-y.mean())).sum() / ((x-x.mean())**2).sum()
c = y.mean() - m*x.mean()

print(m, c)
