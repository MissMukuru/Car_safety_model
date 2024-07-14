The code starts off with importing the various libraries that will be continually used in the
model

we assume the data.info() function to get the basic information in the data which includes things like
a)how many columns and rows we have in the data
b)If we have any missing or null valaues
c)The total size of the data

The next thing we do is give the columns descriptive names 

data['class'].value_counts()-We use this code to know the nature of the column 

X = data . drop(['class'], axis = 1)
X[0:5]

#declaring the target variable 
Y = data["class"]
Y
This code is used to seprate the feature variables from the target variable which in case is the class
This is so because we are trying to figure out 

The next thing we do is essentially split the data into the training set and the test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.33, random_state= 42)
X_train.shape
X_test.shape

This code explains the code used to encode categorical variables in a machine learning dataset using Ordinal Encoding from the category_encoders library. Encoding is crucial for transforming non-numerical categorical data (like "buying" values: "high," "med," "low") 
into numerical representations usable by machine learning algorithms.
#Encoding the variables with ordinal encoding
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
Ordinal_Encoder = ce.OrdinalEncoder(cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = Ordinal_Encoder.fit_transform(X_train)
X_test = Ordinal_Encoder.fit_transform(X_test)
X_train.head(10)


We then move on to training the model using the gini index
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=0 )
clf_gini.fit(X_train,Y_train)
#max_depth = limits the maximum depth of the tree helping to control overfitting
#random_state_0 = ensures that the result are reproducible buy setting the seed for thr random number generator

Then we predict the test results with the gini index
Y_pred_gini = clf_gini.predict(X_test)

Checking the accuracy sore with the criterion gini index
#This means that were using the accuracy score to predict the models performance by comparing the predicted values(Y_pred_gini) with the actual values
from sklearn.metrics import accuracy_score
print("Model accuaracy score with criterion gini index: {0:0.4F}".format(accuracy_score(Y_test,Y_pred_gini)))

COMPARING THE TRAINING SET AND THE TEST SET ACCURACY SCORE TO SEE IF THERE ARE ANY SIGNS OF OVERFITTING

from sklearn.metrics import accuracy_score
Y_train_Pred_gini = clf_gini.predict(X_train)
Y_train_Pred_gini
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(Y_train, Y_train_Pred_gini)))
Training-set accuracy score: 0.7848

Finally we visualize the Tree
from sklearn import tree
plt.figure(figsize=(10,6))
tree.plot_tree(clf_gini.fit(X_train,Y_train))

