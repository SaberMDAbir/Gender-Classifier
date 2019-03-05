from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

'''
This uses four classifiers in the sklearn package
and compares the accruacy result. Prints out the best performing 
classifier based on the dataset. 

Author: MD Abir A. Choudhury - SaberMDAbir
'''

# define dataset programatically in a list --> X and Y
# size of dataset is 11
# [height, weight, shoe size]
X = [
    [181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],
    [175,64,39],[177,70,40],[159,60,40],[171,75,42],[181,85,43]
]

Y = [
    'male','female','female','female','male','male','male','female', 'male',
    'female','male'
]

# split the dataset --> To avoid bias and reduce test error
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size = 0.3,
    random_state = 42
)

# ------------------------------------------------------------------------------
# Decision tree
# Stores our decision tree classifer model in classifier (a tree variable)
classifier_Tree = tree.DecisionTreeClassifier()

# trains the decision tree based on our data
classifier_Tree = classifier_Tree.fit(X_train, y_train)

# test the decision tree and store it in prediction based on the following
# three values inputted in predict
prediction_tree = classifier_Tree.predict([[190,100,43]])

print(prediction_tree)
classifier_Tree = classifier_Tree.score(X_test, y_test)
print(classifier_Tree)
# ------------------------------------------------------------------------------
# SVC --> Support Vector Classification
# stores the svc model to this variable
classifier_svc = SVC(gamma='auto')

# Train (fit) the svc model based on our data
classifier_svc = classifier_svc.fit(X_train,y_train)
# estimator
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

# Test the svc model on the following dataset
prediction_svc = classifier_svc.predict([[190,100,43]])

print(prediction_svc)
classifier_svc = classifier_svc.score(X_test, y_test)
print(classifier_svc)
#-------------------------------------------------------------------------------
# MLPClassifier --> Multi-layer Perceptron Classifier
# stores the mlp model to this variable
classifier_mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=400,
alpha=0.0001, solver='sgd',verbose=10,random_state=21,tol=0.00000000001)
  
classifier_mlp = classifier_mlp.fit(X_train,y_train)
prediction_mlp = classifier_mlp.predict([[190,100,43]])

print(prediction_mlp)
classifier_mlp = classifier_mlp.score(X_test, y_test)
print(classifier_mlp)
#-------------------------------------------------------------------------------
# KNeighborsClassifier --> Nearest Neighbors Classifier
# Classifier implementing the k-nearest neighbors vote
classifier_knn = KNeighborsClassifier(n_neighbors=5)
classifier_knn.fit(X_train,y_train)
predicition_knn = classifier_knn.predict([[190,100,43]])

print(predicition_knn)
classifier_knn = classifier_knn.score(X_test,y_test)
print(classifier_knn)
#-------------------------------------------------------------------------------
# print out the best results
# Input the values in to a dictionary
classifiers = {
    'Tree': [classifier_Tree, prediction_tree],
    'SVC': [classifier_svc, prediction_svc],
    'MLP': [classifier_mlp, prediction_mlp],
    'KNN': [classifier_knn, predicition_knn]
}
# Get the max value in the dictionary
maximum = max(classifiers, key = classifiers.get) 
print("Highest performing classifier is:", maximum, " with a score of ", 
    classifiers[maximum][0], "and a prediction of", *classifiers[maximum][1])