import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import tree
import argparse

class DT:
    """
    A Decision Tree (DT) class for both regression and classification tasks.
    Attributes:
        root (DT.Node): The root node of the decision tree.
        minDataPoints (int): The minimum number of data points required to split a node.
        regressionTree (bool): A flag indicating whether the tree is a regression tree (True) or a classification tree (False).
    Methods:
        fit(x, y):
            Fits the decision tree to the provided data.
        predict(x):
            Predicts the target value for the given input data.
        print():
            Prints the structure of the decision tree.
        _predict(node, x):
            Recursively predicts the target value for the given input data starting from the specified node.
        _print(node, indent):
            Recursively prints the structure of the decision tree starting from the specified node.
        _split(x, y, node, minDataPoints, regressionTree):
            Recursively splits the data to build the decision tree.
        _findBestSplit(x, y, regressionTree):
            Finds the best feature and split point to split the data.
        _calculateRegressionLoss(x, y):
            Calculates the regression loss for potential splits.
        _gini(y):
            Calculates the Gini impurity for the given target values.
        _calculateClassificationGiniLoss(x, y):
            Calculates the Gini loss for potential splits in classification tasks.
        A class representing a node in the decision tree.
        Attributes:
            xIndex (int): The index of the feature used for splitting at this node.
            xSplitPoint (float): The value of the feature used for splitting at this node.
            value (float): The predicted value at this node.
            leftNode (DT.Node): The left child node.
            rightNode (DT.Node): The right child node.
            mse (float): The mean squared error (for regression) or Gini impurity (for classification) at this node.
            nSamples (int): The number of samples at this node.
    """
    class Node:
        xIndex = None
        xSplitPoint = None
        value = None
        leftNode = None
        rightNode = None
        mse = None
        nSamples = None
        
    def __init__(self, minDataPoints=2, regressionTree=True):
        self.root = DT.Node()
        self.minDataPoints = minDataPoints
        self.regressionTree = regressionTree
    
    def fit(self, x, y):
        self._split(x, y, self.root, self.minDataPoints, self.regressionTree)
    
    def predict(self, x):
        return DT._predict(self.root, x)
        
    def print(self):
        DT._print(self.root,"")
        
    @staticmethod 
    def _predict(node, x):
        if node.leftNode is None:
            return node.value
        elif (x[node.xIndex] <= node.xSplitPoint):
            return DT._predict(node.leftNode, x)
        else:
            return DT._predict(node.rightNode, x)
    
    @staticmethod
    def _print(node, indent):
        if node.leftNode is None:
            print(f"{indent}value={node.value:.3f}, mse={node.mse:.3f}, nsamples={node.nSamples}")
        else:
            print(f"{indent}x[{node.xIndex}]={node.xSplitPoint:.3f}, value={node.value:.3f}, mse={node.mse:.3f}, nsamples={node.nSamples}")
            indent += " "
            DT._print(node.leftNode, indent)
            DT._print(node.rightNode, indent)
        
    
    @staticmethod
    def _split(x, y, node, minDataPoints, regressionTree):
        if regressionTree:
            node.value = np.mean(y)
            node.mse = np.sum(np.square(y - node.value))/y.shape[0]
        else:
            ncount = Counter(y)
            node.value = float(ncount.most_common(1)[0][0])
            node.mse = DT._gini(y)
            
        node.nSamples = y.shape[0]
        if not regressionTree and node.mse == 0:
            return
        
        if y.shape[0] < minDataPoints:
            return
        else:
            (x_index, x_point) = DT._findBestSplit(x, y, regressionTree)
            mask = x[:, x_index] <= x_point
            y_left = y[mask]
            y_right = y[~mask]
            x_left = x[mask]
            x_right = x[~mask]
            if y_left.shape[0] == 0 or y_right.shape[0] == 0:
                return
            node.xIndex = x_index
            node.xSplitPoint = x_point
            node.leftNode = DT.Node()
            node.rightNode = DT.Node()
            DT._split(x_left, y_left, node.leftNode, minDataPoints, regressionTree)
            DT._split(x_right, y_right, node.rightNode, minDataPoints, regressionTree)
    
    @staticmethod
    def _findBestSplit(x, y, regressionTree):
        nx = x.shape[1]
        bestXIndex = -1
        bestSplitPoint = -1
        bestLoss = float("inf")
        # print (x.shape, y.shape)
        for i in range(nx):
            xSortNdx = np.argsort(x[:,i])
            xSort = x[xSortNdx,i]
            ySort = y[xSortNdx,]
            # print (xSort.shape, ySort.shape)
            if regressionTree:
                z = DT._calculateRegressionLoss( xSort, ySort)
            else:
                z = DT._calculateClassificationGiniLoss( xSort, ySort)
            l = np.min(z)
            # print (len(xSort))
            # print (z)
            if l < bestLoss:
                argmin_z = np.argmin(z)
                if argmin_z + 1 < len(xSort):
                    bestSplitPoint = (xSort[argmin_z] + xSort[argmin_z + 1]) / 2.0
                else:
                    bestSplitPoint = xSort[argmin_z]
                bestLoss = l
                bestXIndex = i
            
        return (bestXIndex, bestSplitPoint)
    
    @staticmethod
    def _calculateRegressionLoss(x, y):
        """
        """
        N = x.shape[0]
        l = np.zeros(N-1)
        for i in range(1, N):
            mean1 = np.mean(y[0:i,])
            mean2 = np.mean(y[i:,])
            l1 = np.sum(np.square(y[0:i,] - mean1))
            l2 = np.sum(np.square(y[i:,] - mean2))
            l[i-1] = l1 + l2
        return l
    
    @staticmethod
    def _gini(y):
        classes = np.unique(y)
        g = 0
        N = len(y)
        for c in classes:
            p = len(y[y==c])/N
            g += p * (1 - p)
        return g
        
    @staticmethod
    def _calculateClassificationGiniLoss(x, y):
        """
        Given a sorted x returns a vector of loss if split by x_i
        """
        N = x.shape[0]
        l = np.zeros(N-1)
        for i in range(1, N):
            gini1 = DT._gini(y[0:i])
            gini2 = DT._gini(y[i:])
            l[i-1] = gini1 + gini2
        return l

def main(nsize:int, yscaler:float):
    """
    Main function to demonstrate the usage of a custom decision tree (DT) and compare it with sklearn's DecisionTreeRegressor and DecisionTreeClassifier.
    The function performs the following steps:
    1. Generates test data for regression and plots it.
    2. Fits a custom decision tree (DT) to the regression data and prints the tree structure.
    3. Predicts a value using the custom decision tree and prints the result.
    4. Fits sklearn's DecisionTreeRegressor to the same data, plots the tree, and prints the prediction.
    5. Generates test data for classification and plots it.
    6. Fits a custom decision tree (DT) for classification and prints the tree structure.
    7. Predicts a value using the custom decision tree for classification and prints the result.
    8. Fits sklearn's DecisionTreeClassifier to the same data, plots the tree, and prints the prediction.
    """
    # Generate some test data and test the tree
    np.random.seed(0)
    x = np.random.uniform(size=(nsize,1))
    print (x.tolist())
    y = x + np.random.randn(nsize,1)*yscaler
    plt.figure(figsize=(9,5))
    plt.plot(x,y, 'ro')
    plt.show()

    # Fit the tree
    dt = DT()
    dt.fit(x,y)

    # Print the tree
    print ("Printing the custom tree!")
    dt.print()

    # Predict the tree
    print ("Printing the custom prediction!")
    print(f"Predict={dt.predict(np.array([0.5])):.4}")

    # Now lets compare with tree implementation of sklearn
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(x, y)
    print ("Plotting scikit tree")
    tree.plot_tree(clf)
    plt.show()
    print(f"Predict={clf.predict(np.array([0.5]).reshape(1,1))}")

    # Generate some test data and test the tree for classification
    np.random.seed(0)
    x= np.random.uniform(size=(nsize,1))
    y = np.array([ 1 if xi >= 0.5 else 0 for xi in x])
    plt.figure(figsize=(9,5))
    print ("Plotting binary labeled data")
    plt.plot(x,y, 'ro')
    plt.show()

    # Fit the tree
    dtc = DT(regressionTree=False)
    dtc.fit(x,y)

    # Print the tree
    print ("Plotting custom plot!")
    dtc.print()

    # Predict the tree
    print(f"Predict={dtc.predict(np.array([0.6]))}")

    # Compare with sklearn implementation
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf = clf.fit(x, y)
    tree.plot_tree(clf)
    print ("Plotting scikit classification plot!")
    plt.show()
    print(f"Predict={clf.predict(np.array([0.6]).reshape(1,1))}")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Decision Tree Example")
    parser.add_argument('--nsize', type=int, default=10, help='Number of data points')
    parser.add_argument('--yscaler', type=float, default=0.1, help='Scaler for y values')
    args = parser.parse_args()

    nsize = args.nsize
    yscaler = args.yscaler
    main(nsize, yscaler)
