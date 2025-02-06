"""
This script performs image compression using Singular Value Decomposition (SVD).

Modules:
    os: Provides a way of using operating system dependent functionality.
    numpy: Supports large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
    PIL (Python Imaging Library): Adds image processing capabilities to your Python interpreter.
    matplotlib.pyplot: Provides a MATLAB-like plotting framework.

Magic Commands:
    %matplotlib inline: Ensures that matplotlib plots are displayed inline within Jupyter notebooks.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
%matplotlib inline

path = 'waterfall.jpg'
"""
This script opens an image file, prints its dimensions, and displays the image using matplotlib.

Variables:
    path (str): The file path to the image.
    img (PIL.Image.Image): The image object opened from the specified path.

Functions:
    Image.open(path): Opens and identifies the given image file.
    print(): Prints the dimensions of the image.
    plt.title(): Sets the title of the plot.
    plt.imshow(): Displays the image.

Usage:
    Ensure that the specified image file exists at the given path before running the script.
"""
img = Image.open(path)
print("Size(dimension): ",img.size)
plt.title("Original Image")
plt.imshow(img)

# %%
img_grayscale = img.convert('LA')
"""
Converts an image to grayscale, displays it with a title, and saves the grayscale image to a file.

Steps:
1. Convert the input image to grayscale using the 'LA' mode.
2. Display the grayscale image with the title "Image after converting to grayscale".
3. Save the grayscale image to a file named 'dog_grayscale.png'.

Variables:
img_grayscale : PIL.Image.Image
    The grayscale version of the input image.
"""
plt.title("Image after converting to grayscale")
plt.imshow(img_grayscale)
plt.savefig('dog_grayscale.png')

# %%
imgmat = np.array( list(img_grayscale.getdata(band = 0)), float)
"""
This script performs the following operations on a grayscale image:

"""
imgmat.shape = (img_grayscale.size[1], img_grayscale.size[0])
imgmat = np.matrix(imgmat)
imgmat.shape
A = imgmat
A.shape

# %% [markdown]
# ## Computing U, V and S for Singular Value Decomposition

# %% [markdown]
# In Linear Algebra, the Singular Value Decomposition of an (m x n) real or complex matrix (let's say A) is a factorization of the form A = USV^(T) [ S = Sigma ], where U is an (m x m) real or complex unitary matrix, Sigma is an (m x n) rectangular diagonal matrix with non-negative real numbers on the diagonal, and V is an (n x n) real or complex unitary matrix. If A is real, U and V^(T) = V^(*) are real orthogonal matrices.

# %% [markdown]
# The diagonal entries of Sigma are known as the Singular Values of A. The number of non-zero singular values is equal to the rank of A. The columns of U and the columns of V are called the left-singular vectors and right-singular vectors of A, respectively.

# %% [markdown]
# In Singular Value Decomposition we are basically studying our matirx using the most fundamental components that is the eigenvectors and eigenvalues, one advantage being that the matrix need not necessarily be a square matrix unlike PCA.

# %%
Ui = A.dot(A.transpose())
"""
This code snippet calculates the dot product of matrix A and its transpose, 
and then retrieves the shape of the resulting matrix.

- `Ui = A.dot(A.transpose())`: Computes the dot product of matrix A and its transpose.
- `Ui.shape`: Returns the shape (dimensions) of the resulting matrix Ui.

Returns:
    tuple: The shape of the matrix Ui.
"""
Ui.shape

# %%
Vi = A.transpose().dot(A)
"""
Calculate the dot product of the transpose of matrix A with matrix A and return the shape of the resulting matrix.

Variables:
    A (numpy.ndarray): The input matrix.

Returns:
    tuple: The shape of the resulting matrix after the dot product.
"""
Vi.shape

# %% [markdown]
# U here is the eigen vector matrix of AA^(T).
# 
# V here is the eigen vector matrix of A^(T)A.
# 
# U and V both give the same eigen values.
# 
# Sigma or S is a rectangular diagonal matrix that contains the sqaure root of the eigen values of U(or V) which are known as the singular values of A.
# 
# Since A^(T)A and AA^(T) are both square matrices hence eigendecomposition can be easily performed on them to generate the eigen values.

# %%
eig_values, U = np.linalg.eig(Ui)
"""
Performs eigenvalue decomposition on the matrix `Ui` and converts the resulting eigenvalues and eigenvectors to real values.

Steps:
1. Computes the eigenvalues and eigenvectors of the matrix `Ui` using `np.linalg.eig`.
2. Converts the eigenvectors (`U`) to their real parts.
3. Converts the eigenvalues to their real parts.

Returns:
    eig_values (ndarray): The real parts of the eigenvalues of `Ui`.
    U (ndarray): The real parts of the eigenvectors of `Ui`.
    U.shape (tuple): The shape of the matrix `U`.
"""
U = U.real
eig_values.real  #Converting the complex values to real values
U.shape

# %%
V_eig_values, V = np.linalg.eig(Vi)
"""
Performs eigen decomposition on the matrix `Vi` and converts the resulting eigenvalues and eigenvectors to real values.

Variables:
    V_eig_values (ndarray): The eigenvalues of the matrix `Vi`.
    V (ndarray): The eigenvectors of the matrix `Vi`, converted to real values.

Returns:
    tuple: A tuple containing the shape of the matrix `V`.
"""
V = V.real
V_eig_values.real  #Converting the complex values to real values
V.shape

# %%
Si = np.diag(np.sqrt(eig_values[:min(A.shape)]))  # Diagonal matrix with the square root of the eigen values
"""
This code snippet performs the following operations:

1. Creates a diagonal matrix `Si` with the square root of the eigenvalues of matrix `A`.
2. Initializes a rectangular matrix `S` with the same shape as matrix `A`, filled with zeros.
3. Copies the values from `Si` into the top-left corner of `S` to form the Sigma matrix used in Singular Value Decomposition (SVD).

Variables:
- Si: Diagonal matrix containing the square root of the eigenvalues of `A`.
- S: Rectangular matrix that will contain the Sigma matrix for SVD.
"""
S = np.zeros((A.shape[0], A.shape[1]))  # Creating a rectangular matrix that'll act as the base for the actual Sigma matrix
S[:Si.shape[0], :Si.shape[1]] = Si  # Creating the Sigma (or S) matrix
S.shape

# %%
for i in range(0,500,100):
    """
    This code snippet performs image compression using Singular Value Decomposition (SVD) and visualizes the compressed images.

    The loop iterates over a range of values from 0 to 500 with a step of 100. For each iteration:
    1. It computes the compressed image using the first 'i' singular values and vectors.
    2. It displays the compressed image using matplotlib's imshow function with a grayscale colormap.
    3. It sets the title of the plot to indicate the number of singular values used in the compression.
    4. It shows the plot.

    Variables:
    - U: The left singular vectors of the original image matrix.
    - S: The diagonal matrix of singular values of the original image matrix.
    - V: The right singular vectors of the original image matrix.
    - cmp_img: The compressed image matrix.
    - i: The current number of singular values used for compression.
    - title: The title of the plot indicating the number of singular values used.
    """
    cmp_img = np.dot(U[:,:i],(np.dot(S[:i,:i],V[:,:i].transpose())))
    plt.imshow(cmp_img, cmap = 'gray')
    title = " Image after =  %s" %i
    plt.title(title)
    plt.show()

# %% [markdown]
# The quality increases as we increase the total number of components or singluar values but the resolution or quality is still less than the original grayscale image i.e. the image has been compressed.


