import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings('ignore')

# Read CSV
path = "heart.csv"
df = pd.read_csv(path)


dataset = pd.get_dummies(df, columns = ['sex', 'cp', 
                                        'fbs','restecg', 
                                        'exang', 'slope', 
                                        'ca', 'thal'])

standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


X = dataset.drop(['target'], axis = 1)
y = dataset['target']

# knn_scores = []
# for k in range(1,21):
#     knn_classifier = KNeighborsClassifier(n_neighbors = k)
#     score=cross_val_score(knn_classifier,X,y,cv=10)
#     knn_scores.append(score.mean())
#
# plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
# for i in range(1,21):
#     plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
# plt.xticks([i for i in range(1, 21)])
# plt.xlabel('Number of Neighbors (K)')
# plt.ylabel('Scores')
# plt.title('K Neighbors Classifier scores for different K values')
# plt.show()

# KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,X,y,cv=10)
score.mean()

# RandomForestClassifier
randomforest_classifier= RandomForestClassifier(n_estimators=10)
score=cross_val_score(randomforest_classifier,X,y,cv=10)
score.mean()