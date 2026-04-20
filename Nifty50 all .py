#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


# In[2]:


df = pd.read_csv(r"C:\Users\saptu\OneDrive\Documents\Nifty project\NIFTY50_all.csv")
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df['Date'] = pd.to_datetime(df['Date']) #convert to date time 
df = df.sort_values('Date')#ascending order, oldest to newest 
df.head(-5)


# In[5]:


df.sample(5)


# In[6]:


df['MA_5'] = df['Close'].rolling(5).mean().shift(1) #moving avg feature engineering 
df['MA_10'] = df['Close'].rolling(10).mean().shift(1) # considering mean of the last 5 days and 10 days 

#Applying lag functions
df['lag_1'] = df['Close'].shift(1)
df['lag_2'] = df['Close'].shift(2)
df['lag_3'] = df['Close'].shift(3)

df=df.dropna()


# In[7]:


df['volatility'] = df['Close'].rolling(5).std().shift(1)

df.dropna()


# In[8]:


#feaures selection for training 
features = ['MA_5', 'MA_10', 'lag_1', 'lag_2', 'lag_3', 'volatility', 'Volume']
X = df[features] #independent 
y = df['Close'] #dependent 


# In[9]:


tscv = TimeSeriesSplit(n_splits=5)


# In[10]:


#calling XGBoost Reggressor 
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)


# In[11]:


# training + testing loop 
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train) #training, learning patterns 
    preds = model.predict(X_test) #testing, predicting
     # iterations
#evaluateion matrices
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    last_y_test = y_test
    last_preds = preds

print("Final RMSE:", rmse)
print("Final MAE:", mae)


# In[12]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, preds)
print("R2 Score:", r2)


# #Actual vs predicted 

# In[13]:


import matplotlib.pyplot as plt

# choose window size
n = 50  

actual_small = last_y_test.values[-n:]
pred_small = last_preds[-n:]

plt.figure(figsize=(12,6))
plt.plot(actual_small, label='Actual Price')
plt.plot(pred_small, label='Predicted Price')

plt.title("Actual vs Predicted (Last 50 Points)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

plt.show()


# In[14]:


plt.figure(figsize=(6,6))

plt.scatter(actual_small, pred_small)

# Perfect prediction line
plt.plot([actual_small.min(), actual_small.max()],
         [actual_small.min(), actual_small.max()])

plt.title("Actual vs Predicted Scatter")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.show()


# In[17]:


# Step 1: initialize
all_actual = []
all_preds = []

# Step 2: run loop
for train_idx, test_idx in tscv.split(X):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    all_actual.extend(y_test.tolist())
    all_preds.extend(preds.tolist())

# Step 3: convert
all_actual = np.array(all_actual)
all_preds = np.array(all_preds)

# Step 4: sample
n = 150
idx = np.random.choice(len(all_actual), n, replace=False)

actual_sample = all_actual[idx]
pred_sample = all_preds[idx]

# Step 5: plot
plt.figure(figsize=(6,6))
plt.scatter(actual_sample, pred_sample, alpha=0.6)

plt.plot([actual_sample.min(), actual_sample.max()],
         [actual_sample.min(), actual_sample.max()])

plt.title("Actual vs Predicted (Sampled)")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.show()


# In[18]:


df.isnull().sum()


# In[19]:


df.dropna()


# In[20]:


df.isnull().sum()


# In[24]:


# Combine X + y
data = pd.concat([X, y], axis=1)

# Remove all bad values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()


# In[25]:


X = data[features]
y = data['Close']


# In[26]:


print(X.isnull().sum())


# In[27]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse_list = []
mae_list = []
r2_list = []

all_actual = []
all_preds = []

tscv = TimeSeriesSplit(n_splits=5)
model = LinearRegression()

for train_idx, test_idx in tscv.split(X):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Metrices
    rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
    mae_list.append(mean_absolute_error(y_test, preds))
    r2_list.append(r2_score(y_test, preds))

    # Store values (FIXED)
    all_actual.extend(y_test.tolist())
    all_preds.extend(preds.tolist())


# In[28]:


print("===== FINAL RESULTS =====")
print("RMSE:", np.mean(rmse_list))
print("MAE:", np.mean(mae_list))
print("R2 Score:", np.mean(r2_list))


# In[29]:


#Price Trend Over Time
plt.figure(figsize=(12,6))

plt.plot(df['Date'], df['Close'])

plt.title("Stock Price Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")

plt.xticks(rotation=45)
plt.show()


# In[30]:


plt.figure(figsize=(8,5))

plt.hist(df['Close'], bins=30)

plt.title("Distribution of Closing Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.show()


# In[31]:


import seaborn as sns

plt.figure(figsize=(8,6))

corr = df[['Close','MA_5','MA_10','lag_1','lag_2','lag_3','volatility','Volume']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title("Feature Correlation Heatmap")
plt.show()


# In[32]:


plt.figure(figsize=(8,6))

plt.scatter(df['Volume'], df['Close'], alpha=0.5)

plt.title("Volume vs Price")
plt.xlabel("Volume")
plt.ylabel("Close Price")

plt.show()


# In[33]:


plt.figure(figsize=(8,5))

plt.hist(df['volatility'], bins=30)

plt.title("Volatility Distribution")
plt.xlabel("Volatility")
plt.ylabel("Frequency")

plt.show()


# In[34]:


plt.figure(figsize=(12,6))

plt.plot(df['Date'], df['Close'], label='Close')
plt.plot(df['Date'], df['MA_5'], label='MA_5')
plt.plot(df['Date'], df['MA_10'], label='MA_10')

plt.title("Price vs Moving Averages")
plt.legend()

plt.xticks(rotation=45)
plt.show()


# In[35]:


plt.figure(figsize=(6,6))

plt.scatter(df['lag_1'], df['Close'], alpha=0.5)

plt.title("Previous Day Price vs Current Price")
plt.xlabel("Lag 1")
plt.ylabel("Close")

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




