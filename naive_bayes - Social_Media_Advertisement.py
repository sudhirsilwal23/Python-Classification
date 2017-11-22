#################################################################################
#####################  Naive Bayes - Python  ####################################
#################################################################################

#---------------------------------------------------------------------------------
# Step : 1 Importing the libraries
#---------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---------------------------------------------------------------------------------
# Step : 2 Data Preprocessing
#--------------------------------------------------------------------------------
         #2(a) Importing the dataset
dataset = pd.read_csv('Social_Media_Advertisement.csv')
Var_Independent = dataset.iloc[:, [2, 3]].values
Var_dependent   = dataset.iloc[:, 4].values

        #2(b) Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
Var_I_train, Var_I_test, Var_D_train, Var_D_test = train_test_split(Var_Independent, Var_dependent, test_size = 0.25, random_state = 0)

        #2(c) Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Var_I_train = sc.fit_transform(Var_I_train)
Var_I_test = sc.transform(Var_I_test))

#--------------------------------------------------------------------------------
# Step : 3 Data modelling
#--------------------------------------------------------------------------------
        #3(a) Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB
VAR_NB = GaussianNB()
VAR_NB.fit(Var_I_train, Var_D_train)

        #3(b) Predicting the Test set results
Var_D_pred = VAR_NB.predict(Var_I_test)

        #3(c) Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
Var_cm = confusion_matrix(Var_D_test, Var_D_pred)

#--------------------------------------------------------------------------------
# Step : 4 Data Visualising 
#--------------------------------------------------------------------------------
         #4(a) for the Training set results
from matplotlib.colors import ListedColormap
Var_I_set, Var_D_set = Var_I_train, Var_D_train
X1, X2 = np.meshgrid(np.arange(start = Var_I_set[:, 0].min() - 1, stop = Var_I_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = Var_I_set[:, 1].min() - 1, stop = Var_I_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, VAR_NB.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(Var_I_set[Var_D_set == j, 0], Var_I_set[Var_D_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Social Media Advertisement Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

              #4(b) for the Training set results
			  
from matplotlib.colors import ListedColormap
Var_I_set, Var_D_set = Var_I_test, Var_D_test
X1, X2 = np.meshgrid(np.arange(start = Var_I_set[:, 0].min() - 1, stop = Var_I_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = Var_I_set[:, 1].min() - 1, stop = Var_I_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, VAR_NB.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
   plt.scatter(Var_I_set[Var_D_set == j, 0], Var_I_set[Var_D_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Social Media Advertisement Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
