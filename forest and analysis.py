# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:40:15 2023

@author: Deeksha
"""

#Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import scipy.optimize as opt
import seaborn as sns
import plotly.express as px


#importing the dataset
data_countries=pd.read_csv('forest+LAND.csv')
print(data_countries)

#reading continent dataset
data_con=pd.read_csv('countryContinent.csv',encoding='latin')
data_con.sort_values(by='country',ascending=True,inplace=True)
print(data_con)

#considering country dataset
data_countries=data_countries.sort_values(by='Country Name',ascending=True)
print(data_countries)

#merging both country and continent datasets
data=data_countries.merge(data_con,right_on='code_3',left_on='Country Code')
print(data)

#checking for columns
print(data.columns)

#removing unwanted columns
data=data.drop(['Indicator Name','Indicator Code','code_2','code_3','iso_3166_2','country_code','region_code','sub_region_code','country'],axis=1)
print(data)

#checking for missing values
print(data.isnull().sum())

#removing year 1990
data=data.drop(['1990','1991'],axis=1)
print(data)

data=data.dropna(axis=0)
print(data)

#now the data is perfectly cleaned
print(data.isnull().sum())

#creating a column called average
data['average']=data.mean(numeric_only=True,axis=1)
data1=data.copy()
print(data)

datas=pd.DataFrame()
datas=data.copy()
data_new=pd.DataFrame()
data_new['countries']=data['Country Name']

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
data['Country Name']= label_encoder.fit_transform(data['Country Name'])
print(data)

#setting up dataframe that requires columns need to be clustered
data_cluster=pd.DataFrame()
data_cluster['country']=data['Country Name']
data_cluster['average']=data['average']

# set up the clusterer for number of clusters
nclusters=3
kmeans=KMeans(n_clusters=nclusters)

# Fit the data, results are stored in the kmeans object
kmeans.fit(data_cluster) # fit done on x,y pairs

labels = kmeans.labels_
# print(labels) # labels is the number of the associated clusters of (x,y)␣,→points

# extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)

# calculate the silhoutte score
print(skmet.silhouette_score(data_cluster, labels))

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown","tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(nclusters): # loop over the different labels
    plt.plot(data_cluster[labels==l]["country"], data_cluster[labels==l]["average"],"o", markersize=3, color=col[l])
#
# show cluster centres
for ic in range(nclusters):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("Countries")
plt.ylabel("Average")
plt.show()

#for understanding of encoded data of country
data_encoded_countries=pd.DataFrame()
data_encoded_countries['country names']=data_new['countries']
data_encoded_countries['country number']=data['Country Name']
print(data_encoded_countries)

print(datas)

#removing unecessary columns
datas.drop(['Country Code','continent','sub_region','average'],axis=1,inplace=True)
print(datas)

#transposing the dataframe for better use
data_t=datas.T

#converting rows into header column
header_row = 0
data_t.columns = data_t.iloc[header_row]
print(data_t)

# Convert row to column header using DataFrame.iloc[]
data_t.columns = data_t.iloc[0]
print(data_t)

#dropping unwanted columns
data_t.drop('Country Name',axis=0,inplace=True)
print(data_t)

#resetting index values as column name
data_t.reset_index(level=0,inplace=True)
print(data_t)

data_t.rename(columns={'index':'year'},inplace=True)
print(data_t)

#USA,UK,INDIA,CHINA countries are considered and compared
#creating dataframe for UK
df_uk=pd.DataFrame()
df_uk['year']=data_t['year']
df_uk['UK']=data_t['United Kingdom']

#plotting the columns
plt.figure(figsize=(20,20))
df_uk.plot("year","UK")
plt.show()

def exponential(t, n0, g):
#Calculates exponential function with scale factor n0 and growth rate g.
    t = t - 1990.0
    f = n0 * np.exp(g*t)
    return f

print(type(df_uk["year"].iloc[1]))
df_uk["year"] = pd.to_numeric(df_uk["year"])
print(type(df_uk["year"].iloc[1]))
param, covar = opt.curve_fit(exponential, df_uk["year"], df_uk["UK"],p0=(73233967692.102798, 0.03))

df_uk["fit"] = exponential(df_uk["year"], *param)
df_uk.plot("year", ["UK", "fit"])
plt.show()

#creating dataframe for UK
df_usa=pd.DataFrame()
df_usa['year']=data_t['year']
df_usa['Usa']=data_t['United States']

#plotting the columns
plt.figure(figsize=(20,20))
df_usa.plot("year","Usa")
plt.show()

print(type(df_usa["year"].iloc[1]))
df_usa["year"] = pd.to_numeric(df_usa["year"])
print(type(df_usa["year"].iloc[1]))
param, covar = opt.curve_fit(exponential, df_usa["year"], df_usa["Usa"],p0=(73233967692.102798, 0.03))

df_usa["fit"] = exponential(df_usa["year"], *param)
df_usa.plot("year", ["Usa", "fit"])
plt.show()

#creating dataframe for UK
df_ind=pd.DataFrame()
df_ind['year']=data_t['year']
df_ind['india']=data_t['India']

#plotting the columns
plt.figure(figsize=(20,20))
df_ind.plot("year","india")
plt.show()

#print(type(df_usa["year"].iloc[1]))
df_ind["year"] = pd.to_numeric(df_ind["year"])
#print(type(df_usa["year"].iloc[1]))
param, covar = opt.curve_fit(exponential, df_ind["year"], df_ind["india"],p0=(73233967692.102798, 0.03))


df_ind["fit"] = exponential(df_ind["year"], *param)
df_ind.plot("year", ["india", "fit"])
plt.show()

def logistic(t, n0, g, t0):
#"""Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

param, covar = opt.curve_fit(logistic, df_uk["year"], df_uk["UK"],p0=(3e12, 0.03, 2000.0))


sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
df_uk["fit"] = logistic(df_uk["year"], *param)
df_uk.plot("year", ["UK", "fit"])
plt.show()


year = np.arange(1992, 2031)
print(year)
forecast = logistic(year, *param)


plt.figure()
plt.plot(df_uk["year"], df_uk["UK"], label="UK")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("UK")
plt.legend()
plt.show()

param, covar = opt.curve_fit(logistic, df_usa["year"], df_usa["Usa"],p0=(3e12, 0.03, 2000.0))


sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
df_usa["fit"] = logistic(df_usa["year"], *param)
df_usa.plot("year", ["Usa", "fit"])
plt.show()


year = np.arange(1992, 2031)
print(year)
forecast = logistic(year, *param)


plt.figure()
plt.plot(df_usa["year"], df_usa["Usa"], label="Usa")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("Usa")
plt.legend()
plt.show()

param, covar = opt.curve_fit(logistic, df_ind["year"], df_ind["india"],p0=(3e12, 0.03, 2000.0))


sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
df_ind["fit"] = logistic(df_ind["year"], *param)
df_ind.plot("year", ["india", "fit"])
plt.show()


year = np.arange(1992, 2031)
print(year)
forecast = logistic(year, *param)


plt.figure()
plt.plot(df_ind["year"], df_ind["india"], label="india")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("India")
plt.legend()
plt.show()

print(data1)

print(data1['continent'].unique())


def cont(x):
  data_a=data1[data1['continent']==x]
  fig=px.bar(y=data_a['Country Name'],x=data_a['average'],width=1000,height=1200)
  fig.update_layout(title_text=x, title_x=0.5)
  fig.show()
  plt.savefig('BarChart.jpg')

continents=['Asia','Europe','Africa','Americas','Oceania']
for i in continents:
  cont(i)
  
data_mostly=data1[data1['average']>50]
print(data_mostly)

px.bar(y=data_mostly['average'],x=data_mostly['Country Name'],height=800,width=1000)


  
  
  