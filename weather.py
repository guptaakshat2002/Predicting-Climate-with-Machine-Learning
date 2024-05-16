#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[12]:


weather = pd.read_csv("3663781.csv", index_col="DATE")


# In[13]:


weather


# In[14]:


weather.apply(pd.isnull).sum()/weather.shape[0]


# In[17]:


core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]


# In[19]:


core_weather


# In[20]:


core_weather.apply(pd.isnull).sum()


# In[21]:


core_weather["snow"].value_counts()


# In[22]:


del core_weather["snow"]


# In[23]:


core_weather["snow_depth"].value_counts()


# In[24]:


del core_weather["snow_depth"]


# In[25]:


core_weather[pd.isnull(core_weather["precip"])]


# In[26]:


core_weather.loc["1983-10-20":"1983-11-05",:]


# In[27]:


core_weather.loc["2013-12-15",:]


# In[28]:


core_weather["precip"].value_counts() / core_weather.shape[0]


# In[29]:


core_weather["precip"] = core_weather["precip"].fillna(0)


# In[30]:


core_weather.apply(pd.isnull).sum()


# In[31]:


core_weather[pd.isnull(core_weather["temp_min"])]


# In[32]:


core_weather.loc["2011-12-18":"2011-12-28"]


# In[33]:


core_weather = core_weather.fillna(method="ffill")


# In[34]:


core_weather.apply(pd.isnull).sum()


# In[35]:


# Check for missing value defined in data documentation
core_weather.apply(lambda x: (x == 9999).sum())


# In[36]:


core_weather.dtypes


# In[37]:


core_weather.index


# In[38]:


core_weather.index = pd.to_datetime(core_weather.index)


# In[39]:


core_weather.index


# In[40]:


core_weather.index.year


# In[41]:


core_weather.apply(lambda x: (x==9999).sum())


# In[42]:


core_weather[["temp_max", "temp_min"]].plot()


# In[43]:


core_weather.index.year.value_counts().sort_index()


# In[44]:


core_weather["precip"].plot()


# In[45]:


core_weather.groupby(core_weather.index.year).apply(lambda x: x["precip"].sum()).plot()


# In[51]:


core_weather.groupby(core_weather.index.year).sum()


# In[52]:


core_weather.groupby(core_weather.index.year).sum()["precip"]


# In[53]:


core_weather["target"] = core_weather.shift(-1)["temp_max"]


# In[54]:


core_weather


# In[55]:


core_weather = core_weather.iloc[:-1,:].copy()


# In[56]:


core_weather


# In[57]:


from sklearn.linear_model import Ridge


# In[78]:


reg = Ridge(alpha=.3)


# In[79]:


predictors = ["precip", "temp_max", "temp_min"]


# In[94]:


train = core_weather.loc[:"2020-12-31"]


# In[95]:


train


# In[96]:


test = core_weather.loc["2021-01-01":]


# In[97]:


test


# In[98]:


reg.fit(train[predictors], train["target"])


# In[99]:


predictions = reg.predict(test[predictors])


# In[100]:


from sklearn.metrics import mean_squared_error


# In[101]:


mean_squared_error(test["target"], predictions)


# In[102]:


combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]


# In[103]:


combined


# In[104]:


combined.plot()


# In[105]:


reg.coef_


# In[106]:


core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()

core_weather["month_day_max"] = core_weather["month_max"] / core_weather["temp_max"]

core_weather["max_min"] = core_weather["temp_max"] / core_weather["temp_min"]


# In[107]:


core_weather = core_weather.iloc[30:,:].copy()


# In[108]:


def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2020-12-31"]
    test = core_weather.loc["2021-01-01":]

    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])

    error = mean_squared_error(test["target"], predictions)
    
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined


# In[109]:


predictors = ["precip", "temp_max", "temp_min", "month_day_max", "max_min"]

error, combined = create_predictions(predictors, core_weather, reg)
error


# In[110]:


combined.plot()


# In[119]:


core_weather["monthly_avg"] = core_weather["temp_max"].groupby(core_weather.index.month, group_keys=False).apply(lambda x: x.expanding(1).mean())


# In[128]:


core_weather


# In[123]:


core_weather["day_of_year_avg"] = core_weather["temp_max"].groupby(core_weather.index.day_of_year,group_keys= False).apply(lambda x: x.expanding(1).mean())


# In[129]:


predictors = ["precip", "temp_max", "temp_min", "month_max", "month_day_max", "max_min","day_of_year_avg","monthly_avg"]


# In[131]:


error, combine = create_predictions(predictors, core_weather, reg)


# In[132]:


error


# In[133]:


reg.coef_


# In[134]:


core_weather.corr()["target"]


# In[135]:


combined["diff"] = (combined["actual"] - combined["predictions"]).abs()


# In[136]:


combined.sort_values("diff", ascending = False).head()


# In[145]:


# doing prediction
def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2023-12-31"]
    test = core_weather.loc["2024-04-01":]

    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])

    error = mean_squared_error(test["target"], predictions)
    
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined


# In[146]:


predictors = ["precip", "temp_max", "temp_min", "month_day_max", "max_min"]
error, combined = create_predictions(predictors, core_weather, reg)
error


# In[147]:


reg.coef_


# In[148]:


combined.plot()


# In[150]:


core_weather["monthly_avg"] = core_weather["temp_max"].groupby(core_weather.index.month, group_keys=False).apply(lambda x: x.expanding(1).mean())
core_weather["day_of_year_avg"] = core_weather["temp_max"].groupby(core_weather.index.day_of_year,group_keys= False).apply(lambda x: x.expanding(1).mean())


# In[152]:


core_weather


# In[153]:


predictors = ["precip", "temp_max", "temp_min", "month_max", "month_day_max", "max_min","day_of_year_avg","monthly_avg"]
error, combine = create_predictions(predictors, core_weather, reg)
error


# In[154]:


reg.coef_


# In[155]:


core_weather.corr()["target"]


# In[156]:


combined["diff"] = (combined["actual"] - combined["predictions"]).abs()
combined.sort_values("diff", ascending = False).head()


# In[ ]:




