from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
import pandas as pd
import numpy as np

# Loading npy data
X = pd.DataFrame(np.load("PlayingSpiroData/Spiro-Data/npy_file/FEV1_FEATURES_60.npy"))
y = pd.DataFrame(np.load("PlayingSpiroData/Spiro-Data/npy_file/FEV1_LABELS_60.npy"))

# 
# RF = RandomForestRegressor( n_jobs=-1, bootstrap=True, criterion='absolute_error', 
#                                   n_estimators=100,  max_features=1.0, max_depth=300,  
#                                   min_samples_leaf=1, min_samples_split=5 ,random_state=42)

# Instantiating Random forrest Regressor.
RF = RandomForestRegressor()

# DT = DecisionTreeRegressor( criterion='absolute_error',min_samples_leaf=1, 
#                                 min_samples_split=5 ,random_state=42)

# Instantiating Decision Tree Regressor.
DT = DecisionTreeRegressor() 

# Dictionary to specify Estimators/Regressors and their Hyperparameters.
RegDict =   { "Random_Forrest_Regression" :   {"estimator":RF , "param_grid" : {  'bootstrap': [True],
                                                                                'criterion': ['absolute_error'],
                                                                                'max_depth': [50, 100, 150, 300],
                                                                                'min_samples_leaf': [1],
                                                                                'min_samples_split': [5],
                                                                                'n_estimators': [25, 50, 75, 100],
                                                                                'random_state' : [42]
                                                                            }},
                                                                   
             "Decision_Tree_Regression" :    {"estimator":DT , "param_grid" : {  'criterion': ['absolute_error'],
                                                                                'max_depth': [50, 100, 150, 300],
                                                                                'min_samples_leaf': [2],
                                                                                'min_samples_split': [5],
                                                                                'random_state' : [42]
                                                                            }},
            }

abserror = []
tot = len(X)

# Instantiating Leave_One_Out split function.
loo = LeaveOneOut()

for Reg in RegDict:

    print("=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("For : ",Reg)
    Model = RegDict[Reg]['estimator']

    prog = 0
    y_GT = []
    y_PT = []

    for i,(train_index, test_index) in enumerate(loo.split(X)):

        prog = prog + 1
        print("Progress : {0}/{1}".format(prog,tot))

        X_Train, X_Test = X.iloc[train_index],X.iloc[test_index]
        y_Train, y_Test = y.iloc[train_index],y.iloc[test_index]
        
        # Performing Cross Validation to estimate the best estimator given the parameters in param_grid
        grid_search = GridSearchCV(estimator = RegDict[Reg]["estimator"], param_grid = RegDict[Reg]["param_grid"], 
                                    cv = 5 , verbose = 3 , scoring = 'neg_mean_absolute_percentage_error')
        grid_search.fit(X_Train, np.ravel(y_Train))
        Model = grid_search.best_estimator_

        # Model.fit(X_Train, np.ravel(y_Train))
        pred = Model.predict(X_Test)   
        y_GT.append(y_Test.iloc[0,0])
        y_PT.append(pred[0])  

        abserror.append(np.abs( (y_Test.iloc[0,0] - pred[0])/ y_Test.iloc[0,0] ))

    # Calculating Metrics
    MAPE = 100 * np.mean(abserror)
    print("Calculated MAPE : ", MAPE )
    print(" sklearm MAPE : " , 100 * mean_absolute_percentage_error(y_GT,y_PT))
    print(" sklearm MAE : " , mean_absolute_error(y_GT,y_PT))
    print(" sklearm MSE : " , mean_squared_error(y_GT,y_PT))





    