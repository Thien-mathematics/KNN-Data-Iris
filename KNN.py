import numpy as np
from numpy import genfromtxt

# Importing the dataset
data = genfromtxt("/content/iris_full.csv", delimiter=",", skip_header=1, )
#Shuffle the data
np.random.shuffle(data)
#Data 
X_train = data[:80,:4]
y_train = data[:80,4:]
X_test = data[80:,:4]
y_test = data[80:,4:]
#Function to calculate the euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
#Function to calculate the KNN
def KNN(X_train, y_train, X_test, k=3):
    y_pred = np.zeros((len(X_test), 1))
    for i in range(len(X_test)):
        distances = [euclidean_distance(X_test[i], x) for x in X_train]
        k_idx = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_idx]
        y_pred[i] = max(k_nearest_labels, key=k_nearest_labels.count)
    return y_pred
#Vote function
def vote(y):
    return max(y, key=y.count)
#Accuracy function
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#Test the model
y_pred = KNN(X_train, y_train, X_test, k=3)
print("Accuracy:", accuracy(y_test, y_pred))
