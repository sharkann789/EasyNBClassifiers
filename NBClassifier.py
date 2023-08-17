from numpy import *

"""
Sharath Kannan
-
A simple Naive Bayes Classifier.
X is an array of all feature vectors.
y is an array of all classes respective to feature vectors.
"""

class NBClassifier:
    X = []
    y = []

    def __init__(self, X, y):
        self.X = X
        self.y = y


    # set  X and y to this
    def train(self, X, y):
        self.X = X
        self.y = y

    # Xpr if a feature vector.
    def predict(self, Xpr):
        # collection of unique classes.
        yset = set(self.y)

        # final list where we check for max.
        maxcheck = []
        
        # for each class
        # if continuous, use gaussian. If not, use regular!
        for yc in yset:

            # P(c)
            pc = list(self.y).count(yc) / len(self.y) 

            #length of inprod should be the number of classes.
            indprod = []

            for i in range (0, len(Xpr)):
                # grab the column for values for a particular feature value that corresponds to yc.
                ind = where(self.y == yc)
                col = (self.X[:,i])[ind]
                
                # P(x|c) - Gaussian conditional probability.
                f = 1 / sqrt(2 * pi * var(col))
                s = exp( -1 * (square(Xpr[i] - mean(col)) / (2 * var(col))) )
                indprod.append(f*s)
            
            #product of all conditional probabilities for each feature value
            l = prod(indprod)
            
            # multiply with P(c) and append to list.
            maxcheck.append( pc * l )

        # return the class with the biggest probability.
        return argmax(maxcheck)