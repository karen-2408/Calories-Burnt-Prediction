import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# load the dataset into the panda's data frame
df = pd.read_csv('calories.csv')
df.head()

# check the size of the dataset.
df.shape

# check which column of the dataset contains which type of data.
df.info()

# check the descriptive statistical measures of the data.
df.describe()

# to check the data analysis by seeing the scatterplot
sb.scatterplot(x='Height', y='Weight', data=df) 
plt.show()
#shows a linear relationship 

features = ['Age', 'Height', 'Weight', 'Duration']
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    x = df.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=x)
plt.tight_layout()
plt.show()
# higher is the duration of the workout higher will be the calories burnt.
