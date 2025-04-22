#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
arshmankhalid_shopify_streaming_history_dataset_path = kagglehub.dataset_download('arshmankhalid/shopify-streaming-history-dataset')

print('Data source import complete.')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Lets apply machine learning to spotify data
# Applying machine learning to spotify data we can predict if the track will be fully played or skipped by using the factors like Playtime, platform, shuffle mode and more.

# **Loading Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# **Loading and Preparing the Data**

# In[ ]:


# Load the datasets
spotify_data = pd.read_csv('/kaggle/input/shopify-streaming-history-dataset/spotify_history.csv', encoding='UTF-8-SIG')
data_dictionary = pd.read_csv('/kaggle/input/shopify-streaming-history-dataset/spotify_data_dictionary.csv', encoding='UTF-8-SIG')

# Display the first few rows of the spotify history dataset
spotify_data.head()


# **Converting TS to Datetime and ms played to minutes**

# In[ ]:


#convert the timestamp to datetime
spotify_data['ts']= pd.to_datetime(spotify_data['ts'], utc=True)
#convert ms_played to minutes
spotify_data['minutes_played'] = spotify_data['ms_played']/60000


# **Extracting Time feature**

# In[ ]:


spotify_data ['ts'] = pd.to_datetime(spotify_data['ts'], errors='coerce',utc=True)
spotify_data['hour'] = spotify_data ['ts'].dt.hour
spotify_data['day'] = spotify_data ['ts'].dt.day
spotify_data['month'] = spotify_data ['ts'].dt.month


# **Converting categorical to numerical**

# In[ ]:


encoder = LabelEncoder()
spotify_data['platform']= encoder.fit_transform(spotify_data['platform'])
spotify_data['reason_start']= encoder.fit_transform(spotify_data['reason_start'])
spotify_data['reason_end']= encoder.fit_transform(spotify_data['reason_end'])


# **Convert shuffle and skip to binary**

# In[ ]:


spotify_data['shuffle'] = spotify_data['shuffle'].astype(int)
spotify_data['skipped'] = spotify_data['skipped'].astype(int)


# **Dropping columns which are not usefull**

# In[ ]:


spotify_data.drop(['ts', 'spotify_track_uri', 'track_name', 'artist_name', 'album_name', 'ms_played'], axis=1, inplace=True)


# **Defining X and Y traget Variable**

# In[ ]:


x = spotify_data.drop(columns=['skipped'])
y = spotify_data['skipped']


# **Traning the data Set**

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# **Scalling the numerical feature**

# In[ ]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# **Hyperparameter tuning using grid csv**

# In[ ]:


param_grid = {
    'n_estimators': [50],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Initialize Random Forest model
rf = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(x_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)


# **Traning Final Model**

# In[ ]:


# Train model using best parameters
best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
best_rf.fit(x_train, y_train)

# Predictions
y_pred = best_rf.predict(x_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.2f}")

print("Classification Report:\n", classification_report(y_test, y_pred))


