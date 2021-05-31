import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as sc
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# import keras
from sklearn.preprocessing import StandardScaler, LabelBinarizer


def loadData(location="data/iris.csv"):
    def plotData(iris):
        # plots and visualizes the data
        fig = iris[iris.species == 'setosa'].plot(kind='scatter', x='sepal_length', y='sepal_width', color='orange',
                                                       label='setosa')
        iris[iris.species == 'versicolor'].plot(kind='scatter', x='sepal_length', y='sepal_width', color='blue',
                                                     label='versicolor', ax=fig)
        iris[iris.species == 'virginica'].plot(kind='scatter', x='sepal_length', y='sepal_width', color='green',
                                                    label='virginica', ax=fig)
        fig.set_xlabel("Sepal Length")
        fig.set_ylabel("Sepal Width")
        fig.set_title("Sepal Length VS Width")
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        plt.savefig("figures/Sepal Length VS Width.jpg")
        fig = iris[iris.species == 'setosa'].plot.scatter(x='petal_length', y='petal_width', color='orange',
                                                               label='setosa')
        iris[iris.species == 'versicolor'].plot.scatter(x='petal_length', y='petal_width', color='blue',
                                                             label='versicolor', ax=fig)
        iris[iris.species == 'virginica'].plot.scatter(x='petal_length', y='petal_width', color='green',
                                                            label='virginica', ax=fig)
        fig.set_xlabel("Petal Length")
        fig.set_ylabel("Petal Width")
        fig.set_title("Petal Length VS Width")
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        plt. savefig("figures/Petal Length VS Width.jpg")
        iris.hist(edgecolor='black', linewidth=1.2)
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        plt.savefig("figures/Sepal and Petal features")
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        sns.violinplot(x='species', y='petal_length', data=iris)
        plt.subplot(2, 2, 2)
        sns.violinplot(x='species', y='petal_width', data=iris)
        plt.subplot(2, 2, 3)
        sns.violinplot(x='species', y='sepal_length', data=iris)
        plt.subplot(2, 2, 4)
        sns.violinplot(x='species', y='sepal_width', data=iris)
        plt.savefig("figures/species features")
        plt.figure(figsize=(10, 8))
        sns.heatmap(iris.corr(), annot=True,
                    cmap='cubehelix_r')  # draws  heatmap with input as the correlation matrix calculted by(iris.corr())
        plt.savefig("figures/features")
        plt.close('all')
        fig = sns.pairplot(iris, hue='species', markers='+')
        plt.savefig("figures/all data")
        plt.close('all')
    # reads from csv file
    iris = pd.read_csv(location)
    # gets some info about the dataset
    print(iris.info())
    print(iris['species'].value_counts())
    print(iris.describe())
    # visualizes the data
    plotData(iris)
    return iris


def SVM(train_X, train_y, test_X, test_Y):
    # trains SVM model
    print("----------------------------------------------" +"\n" + "SVM: ")
    model = svm.SVC()
    model.fit(train_X, train_y)  # trains the model with training set
    prediction = model.predict(test_X)  # predicts y for both test_X and train_X
    predictionTrain = model.predict(train_X)
    # computes the accuracy of prediction for both train and test set
    print('The accuracy of the SVM on train set is', metrics.accuracy_score(predictionTrain, train_y))
    print('The accuracy of the SVM on the test set is:',metrics.accuracy_score(prediction, test_y))  # now we check the accuracy of the algorithm.
    # prints the predicted y for test set
    testsX = []
    testsY = []
    for index, row in test_X.iterrows():
        test = []
        for i, value in row.items():
            test.append(value)
        testsX.append(test)
    for index, specie in test_Y.items():
        testsY.append(specie)
    for i in range(len(testsX)):
        print(testsX[i])
        print(testsY[i])
        print("prediction: " + prediction[i])


def decisionTree(train_X, train_y, test_X, test_Y):
    # trains a decision tree model
    print("----------------------------------------------"+ "\n" + "DECISION TREE: ")
    model = DecisionTreeClassifier()
    model.fit(train_X, train_y) # trains the model with training set
    prediction = model.predict(test_X) # predicts y for both test_X and train_X
    predictionTrain = model.predict(train_X)
    # computes the accuracy of prediction for both train and test set
    print('The accuracy of the Decision Tree on train set is', metrics.accuracy_score(predictionTrain, train_y))
    print('The accuracy of the Decision Tree on test set is', metrics.accuracy_score(prediction, test_y))
    # prints the predicted y for test set
    testsX = []
    testsY = []
    for index, row in test_X.iterrows():
        test = []
        for i, value in row.items():
            test.append(value)
        testsX.append(test)
    for index, specie in test_Y.items():
        testsY.append(specie)
    for i in range(len(testsX)):
        print(testsX[i])
        print(testsY[i])
        print("prediction: " + prediction[i])

def kNearestNeighbors(train_X, train_y, test_X, test_y,k=3):
    # trains a k- nearest neighbors model
    print("----------------------------------------------" + "\n" + "K-NEAREST NEIGHBORS n= " + str(k) + ": ")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_X, train_y) # trains the model with training set
    prediction = model.predict(test_X) # predicts y for both test_X and train_X
    predictionTrain = model.predict(train_X)
    # computes the accuracy of prediction for both train and test set
    print('The accuracy of KNN on train set is', metrics.accuracy_score(predictionTrain, train_y))
    print('The accuracy of KNN on test set is', metrics.accuracy_score(prediction, test_y))
    # prints the predicted y for test set
    testsX = []
    testsY = []
    for index, row in test_X.iterrows():
        test = []
        for i, value in row.items():
            test.append(value)
        testsX.append(test)
    for index, specie in test_y.items():
        testsY.append(specie)
    for i in range(len(testsX)):
        print(testsX[i])
        print(testsY[i])
        print("prediction: " + prediction[i])


def testkNearestNeighbors(train_X, train_y, test_X, test_y, nmax):
    # finds the best n for k- nearest neighbors algorithm in (1, nmax)
    k_range = list(range(1, nmax))
    scores = []
    for k in k_range: # computes accuracy for each k
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_X, train_y)
        prediction = knn.predict(train_X)
        scores.append(metrics.accuracy_score(train_y, prediction))
    kmax = scores.index(max(scores)) + 1 # finds the best n
    plt.plot(k_range, scores) # plots accuarcy
    plt.xlabel('Value of k for KNN')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
    plt.savefig("figures/Accuracy Scores for Values of k of k-Nearest-Neighbors")
    return kmax


def logisticRegression(train_X, train_y, test_X, test_y):
    # trains a logistic regression model
    print("----------------------------------------------" + "\n" + "LOGISTIC REGRESSION: ")
    model = LogisticRegression()
    # trains the model with training set
    model.fit(train_X, train_y)
    # predicts y for both test_X and train_X
    prediction = model.predict(test_X)
    predictionTrain = model.predict(train_X)
    # computes the accuracy of prediction for both train and test set
    print('The accuracy of the Logistic Regression on train set is', metrics.accuracy_score(predictionTrain, train_y))
    print('The accuracy of the Logistic Regression on test set is', metrics.accuracy_score(prediction, test_y))
    # prints the predicted y for test set
    testsX = []
    testsY = []
    for index, row in test_X.iterrows():
        test = []
        for i, value in row.items():
            test.append(value)
        testsX.append(test)
    for index, specie in test_y.items():
        testsY.append(specie)
    for i in range(len(testsX)):
        print(testsX[i])
        print(testsY[i])
        print("prediction: " + prediction[i])

def deepLearning(train_X, train_y, test_X, test_y):
    # trains a shallow neural network
    shallow_model = keras.models.Sequential()
    shallow_model.add(Dense(4, input_dim=4, activation='relu'))
    shallow_model.add(Dense(units=10, activation='relu'))
    shallow_model.add(Dense(units=3, activation='softmax'))
    shallow_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    shallow_history = shallow_model.fit(train_X, train_y, epochs=150, validation_data=(test_X, test_y))
    plt.close('all')
    plt.plot(shallow_history.history['acc'])
    plt.plot(shallow_history.history['val_acc'])
    plt.title("Accuracy")
    plt.legend(['train', 'test'])
    plt.savefig("figures/shallow learning accuracy")
    plt.close('all')
    plt.plot(shallow_history.history['loss'])
    plt.plot(shallow_history.history['val_loss'])
    plt.plot('Loss')
    plt.legend(['Train', 'Test'])
    plt.savefig("figures/shallow learning loss")







if __name__ == "__main__":
    iris = loadData()
    # splits data into train and test set, with the ratio 0.3
    train, test = sc.train_test_split(iris, test_size=0.3)
    train_X = train[['sepal_length','sepal_width', 'petal_length', 'petal_width']]
    train_y = train.species
    test_X = test[['sepal_length','sepal_width', 'petal_length', 'petal_width']]
    test_y = test.species
    # SVM
    SVM(train_X, train_y, test_X, test_y)
    # Decision Tree
    decisionTree(train_X, train_y, test_X, test_y)
    # Finding the best n for k- nearest neighbors, n is the best k within (1, nmax)
    n = testkNearestNeighbors(train_X, train_y, test_X, test_y, nmax=25)
    # k- nearest neighbors
    kNearestNeighbors(train_X, train_y, test_X, test_y, n)
    # logistic regression
    logisticRegression(train_X, train_y, test_X, test_y)
    # deepLearning(train_X, train_y, test_X, test_y)
