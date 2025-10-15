#%%
import pandas as pd
import numpy as np;
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
#%%
path = Path("lecture/03/sample_stats.csv")
df = pd.read_csv(path).set_index("idx").astype(float)
#%%
df
# %%
for i in range(np.max(df.index)):
  mean = np.mean(df[df.index == i]['y'])
  stdev = np.std(df[df.index == i]['y'])
  print(f"Dataset {i} mean: {mean}")
  print(f"Dataset {i} stdev: {stdev}")
# %%
for i in range(np.max(df.index)):
  plt.scatter(df[df.index == i]['x'], df[df.index == i]['y'])
  plt.show()
#%%
cau = scipy.stats.cauchy.rvs(size=1000)
sm.qqplot(data=cau)
# %%
gau1 = scipy.stats.norm.rvs(loc=3,size=1000)
gau2 = scipy.stats.norm.rvs(loc=2,size=1000)
gau = np.concatenate((gau1,gau2))

sm.qqplot(data=gau)
# %%
