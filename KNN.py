# %%
#Lets use Sklearn to implement KNN
"""
This script demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm using the scikit-learn library.
It imports necessary libraries including numpy for numerical operations, pandas for data manipulation, and matplotlib for plotting.
Imports:
    - numpy as np: For numerical operations.
    - pandas as pd: For data manipulation and analysis.
    - matplotlib.pyplot as plt: For creating static, animated, and interactive visualizations.
The script also enables inline plotting for Jupyter Notebooks using the `%matplotlib inline` magic command.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
from matplotlib.colors import ListedColormap
"""
Plot decision regions for a classifier.
Parameters
----------
X : array-like, shape = [n_samples, n_features]
    Feature matrix.
y : array-like, shape = [n_samples]
    Target values.
classifier : object
    A trained classifier with a `predict` method.
test_idx : array-like, shape = [n_test_samples], optional (default=None)
    Indices of test samples to be highlighted.
resolution : float, optional (default=0.02)
    Resolution of the grid for plotting decision surface.
plotDecisionSurface : bool, optional (default=True)
    Whether to plot the decision surface.
Returns
-------
None
"""
import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02, plotDecisionSurface=True):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    if plotDecisionSurface:
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c=colors[0],
                    edgecolor='black',
                    alpha=1,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

# %%
#import iris dataset
"""
This loads the Iris dataset, selects two features 
(petal length and petal width), splits the dataset into training and testing 
sets while maintaining the class distribution, and prints the counts of each 
class label in the original, training, and testing sets. This preprocessing step 
is essential for preparing the data for training and evaluating a machine learning model.
"""
from sklearn import datasets

iris = datasets.load_iris()
#load only two features
X = iris.data[:,[2,3]]
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_test:', np.bincount(y_test))
print('Labels counts in y_train:', np.bincount(y_train))

# %%
#Standardize the features
"""standardizes the features of the training and testing datasets 
using the StandardScaler class from scikit-learn. It fits the scaler 
to the training data, transforms both the training and testing data, 
and then combines the standardized data into a single feature matrix 
and target vector. Standardization is an essential preprocessing step 
that helps ensure that the features are on a similar scale, which can 
improve the performance of many machine learning algorithms.
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# %%
"""
applies the k-Nearest Neighbors (KNN) algorithm to classify samples in a dataset. 
It imports the KNeighborsClassifier class from scikit-learn, creates an instance 
of the classifier with 3 nearest neighbors, fits the classifier to the standardized 
training data, and makes predictions on both the training and testing data. 
The predicted class labels for the training and testing samples are stored in 
the variables y_gnb_train_pred and y_gnb_test_pred, respectively. 
This process demonstrates how to use KNN for classification tasks in a machine learning workflow.
"""
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
y_gnb_train_pred = neigh.fit(X_train_std, y_train).predict(X_train_std)
y_gnb_test_pred = neigh.predict(X_test_std)

# %%
#Plot the decision boundary
plt.figure(figsize=(20,10))
plot_decision_regions(X_combined_std, y_combined, classifier=neigh, test_idx=range(105,150))
plt.xlabel('petal length [standardized]', fontsize=20)
plt.ylabel('petal width [standardized]', fontsize=20)
plt.legend(fontsize=20)
plt.axis()
plt.title('KNN', fontsize=20)
plt.show()
# %%
#Lets see how the decision boundary changes as we change number of neighbours
"""
In summary, this code snippet visualizes how the decision boundaries 
of a k-Nearest Neighbors (KNN) classifier change as the number of neighbors 
(n_neighbors) varies. It iterates over different values of n_neighbors, 
trains a KNN classifier for each value, and plots the decision boundaries 
using the plot_decision_regions function. The plots include labeled axes, 
a legend, and a title, providing a clear and informative visualization of 
the classifier's behavior with different neighborhood sizes.
"""
neighbours = [1, 2, 3, 4, 5, 10, 50]
for n in neighbours:
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(X_train_std, y_train)
    plt.figure(figsize=(20,10))
    plot_decision_regions(X_combined_std, y_combined, classifier=neigh, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]', fontsize=20)
    plt.ylabel('petal width [standardized]', fontsize=20)
    plt.legend(fontsize=20)
    plt.axis()
    plt.title(f'KNN neighs={n}', fontsize=20)
    plt.show()

# %%
from scipy import stats
"""KNN class implements the k-Nearest Neighbors algorithm with methods for 
fitting the model, calculating distances, and making predictions. 
The distance method calculates the Euclidean distance between points, 
and the predict method uses these distances to find the most common 
class label among the nearest neighbors. The import error can be resolved 
by installing the scipy library."""

class KNN():
    
    def __init__(self, k=3):
        self.k_ = k
        
    def fit(self, X,y):
        self.X_ = X
        self.y_ = y
    
    def distance(self, x1,x2):
        """
            distance of x2 from every point in vector x1
        """
        return np.sum(np.square(x1-x2),axis=1)
    
    def predict(self, X):
        ypred = np.zeros(X.shape[0])
        for ndx, xi in enumerate(X):
            #Find distance of xi from every element in self.X
            dist = self.distance(self.X_, xi)
            #Sort the distance and get the k smallerst 
            smallest_k = np.argsort(dist,kind='stable')[:self.k_]
            ypred[ndx] = stats.mode(self.y_[smallest_k]).mode
        return ypred

# %%
"""demonstrates how to use a custom KNN classifier to classify 
samples in a dataset. An instance of the KNN class is created with k=3, 
the model is fitted to the standardized training data, and predictions 
are made on the standardized test data. The predicted class labels for 
the test samples are stored in the variable y_gnb_test_pred. 
This process showcases the steps involved in training and using a 
KNN classifier for classification tasks."""
myNeigh = KNN(k=3)
myNeigh.fit(X_train_std, y_train)
y_gnb_test_pred = myNeigh.predict(X_test_std)

# %%


# %%
#Plot the decision boundary
plt.figure(figsize=(20,10))
plot_decision_regions(X_combined_std, y_combined, classifier=myNeigh, test_idx=range(105,150))
plt.xlabel('petal length [standardized]', fontsize=20)
plt.ylabel('petal width [standardized]', fontsize=20)
plt.legend(fontsize=20)
plt.axis()
plt.title('KNN', fontsize=20)
plt.show()


