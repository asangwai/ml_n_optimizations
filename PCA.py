# %% [markdown]
# ## PCA
# **PCA** (Principal Component Analysis) is one of the most important unsupervised learning technique. It finds the hyperplane that preserves the maximum amount of variance in the data. Its a common technique to reduce dimensionality of the data, which loosing least possible variance of the data. **PCA** and **SVD** are very closely related. **SVD** is a factorization of a matrix $X=U \Sigma V^T$, where $U$ and $V$ are orthognal matrices. Columns of $V^T$ are principal components of matrix $X$. Let's see the following in examples below.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
#Load IRIS dataset
from sklearn.datasets import load_iris
data = load_iris()
X = data['data']

# %%
from sklearn.preprocessing import StandardScaler
"""
This script standardizes the features of a dataset using the StandardScaler from scikit-learn.

Modules:
    from sklearn.preprocessing import StandardScaler: Imports the StandardScaler class for standardizing features.

Variables:
    sd (StandardScaler): An instance of the StandardScaler class.
    X_std (ndarray): The standardized version of the input dataset X.

Functions:
    sd.fit(X): Computes the mean and standard deviation for scaling based on the input dataset X.
    sd.transform(X): Standardizes the input dataset X using the computed mean and standard deviation.
"""
sd = StandardScaler()
sd.fit(X)
X_std = sd.transform(X) 

# %%
from sklearn.decomposition import PCA
"""
This script performs Principal Component Analysis (PCA) on standardized data.

Modules:
    from sklearn.decomposition import PCA: Imports the PCA class from scikit-learn's decomposition module.

Usage:

Variables:
    pca (PCA): An instance of the PCA class with a fixed random state for reproducibility.
    X_std (array-like): The standardized data to be used for fitting the PCA model.
"""
pca = PCA(random_state=0)
pca.fit(X_std)

# %%
print("Components")
print(pca.components_)
print("-"*50)
print("Singular values")
print(pca.singular_values_)

# %%
#Lets plot variance explained as a function of principal components
"""
This script plots the cumulative variance explained as a function of principal components.

Steps:
1. Calculate the variance explained by each principal component.
2. Compute the cumulative sum of the explained variance.
3. Create a figure with specified dimensions.
4. Add axes to the figure.
5. Plot the cumulative variance explained as a function of principal components.
6. Set the title, x-label, and y-label for the plot.

Variables:
- cv: Array of explained variance ratios for each principal component.
- fig: Matplotlib figure object.
- ax: Matplotlib axes object.

Plot:
- x-axis: Principal Components (PC)
- y-axis: Cumulative fractional variance explained
"""
cv = pca.explained_variance_/pca.explained_variance_.sum()
cv.cumsum()
fig = plt.figure(figsize=(15,10))
ax = fig.add_axes([0,0,1,1])
ax.plot(cv.cumsum(), 'ro')
ax.set_title('Variance explained as a fuction of PC')
ax.set_xlabel('PC')
ax.set_ylabel('Cumulative fractional variance explained')

# %% [markdown]
# We can see from the above plot **two PC directions** can explain most of the variance (>95%) in the data.

# %%
P = pca.components_[0:2,:]
"""
This code snippet performs the following operations:

1. Extracts the first two principal components from the PCA object.
2. Projects the standardized data onto the new PCA components.
3. Displays the first 10 rows of the transformed data.

Variables:
- P: The first two principal components from the PCA object.
- X_std_pca: The standardized data projected onto the first two principal components.

Output:
- The first 10 rows of the transformed data (X_std_pca).
"""
X_std_pca = X_std @ P.T
X_std_pca[:10]

# %% [markdown]
# Now lets plot target as a function of PC1 and PC2.

# %%
df = pd.DataFrame(X_std_pca, columns=['PC1', 'PC2'])
"""
This code snippet performs the following tasks:
1. Creates a DataFrame `df` using the standardized PCA-transformed data `X_std_pca` with columns 'PC1' and 'PC2'.
2. Adds a 'target' column to the DataFrame `df` from the original dataset `data`.
3. Plots a scatter plot using seaborn to visualize the PCA components with different colors for each target class.

Variables:
- X_std_pca: The standardized PCA-transformed data.
- data: The original dataset containing the 'target' column.
- df: The DataFrame containing PCA components and target labels.

Libraries:
- pandas as pd
- matplotlib.pyplot as plt
- seaborn as sns

Plot:
- x-axis: Principal Component 1 (PC1)
- y-axis: Principal Component 2 (PC2)
- hue: Target classes from the original dataset
- palette: Bright colors for different target classes
- figure size: 15x10 inches
"""
df['target'] = data['target']
plt.figure(figsize=(15,10))
sns.scatterplot(x='PC1', y='PC2', data=df, hue='target', palette='bright')

# %% [markdown]
# You can see from above that data is pretty well separated when we plot in PC1 and PC2 directions. Remember **PCA** doesn't look at the targets, it assumes where there is most variation in features there will also be most variation in targets. Next lets compute PCA from scratch using **SVD**.

# %%
(u,s,vh) = np.linalg.svd(X_std)
"""
Performs Singular Value Decomposition (SVD) on the standardized data matrix X_std and prints the principal components and singular values.

Parameters:
None

Returns:
None

Prints:
- Principal components (vh): The right singular vectors of the standardized data matrix.
- Singular values (s): The singular values of the standardized data matrix.
"""
print("Principal components",vh)
print("-"*50)
print("Singular values", s)

# %% [markdown]
# These PCs match as given by **Sklearn** (they are sometimes off by a -1, that doesn't matter because its a constant scaler)

# %% [markdown]
# Finally lets try and get Principal components by **Eigen Value decomposition**. Getting PC by EVD is not a preferred method but is informative.

# %%
#Lets compute XX^T
"""
This code snippet computes the principal components and singular values of a standardized dataset.

Steps:
1. Compute the covariance matrix (XX^T) of the standardized dataset `X_std`.
2. Perform eigen decomposition on the covariance matrix to obtain eigenvalues and eigenvectors.
3. Print the principal components (eigenvectors).
4. Print the singular values (square roots of the eigenvalues).

Variables:
- X_std: Standardized dataset (numpy array).
- C: Covariance matrix of the standardized dataset.
- w: Eigenvalues of the covariance matrix.
- v: Eigenvectors (principal components) of the covariance matrix.
"""
C = X_std.T@X_std
(w, v) = np.linalg.eig(C)
print("Principal components",v)
print("-"*50)
print("Singular values", np.sqrt(w))

# %% [markdown]
# Principal components and signular values calculated this way also match the results from **sklearn** and **SVD**. This calculation shows that **PCA** vectors are principal components of $X^TX$.


