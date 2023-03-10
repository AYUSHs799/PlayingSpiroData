from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
import pandas as pd
import numpy as np

# Loading npy data.
X = np.load("PlayingSpiroData/Spiro-Data/npy_file/FEV1_FEATURES_60.npy")
y = np.load("PlayingSpiroData/Spiro-Data/npy_file/FEV1_LABELS_60.npy")

X= pd.DataFrame(np.delete(X,[23,55,4,9,52,44,45,33,43,20,1,50],axis=0))
y= pd.DataFrame(np.delete(y,[23,55,4,9,52,44,45,33,43,20,1,50],axis=0))

# Instantiating Random forrest Regressor.
RF = RandomForestRegressor( n_jobs=-1, bootstrap=True, criterion='absolute_error', 
                                  n_estimators=500,  max_features=1.0, max_depth=300,  
                                  min_samples_leaf=1, min_samples_split=5 ,random_state=42)

# Instantiating Decision Tree Regressor.
DT = DecisionTreeRegressor( criterion='absolute_error',min_samples_leaf=1, 
                                min_samples_split=5 ,random_state=42)

# Instantiating Linear Regressor.
LR = LinearRegression()

# Estimator/Regressor Dictionary.
RegDict = { "Random_Forrest_Regression" :   RF,
            "Decision_Tree_Regression" :    DT,
            "Linear_Regression":            LR }

tot = len(X)
# Instantiating Leave_One_Out split function.
loo = LeaveOneOut()

def MAPE_func(y_true,y_pred):
    abserr = np.zeros(len(y_true))
    for i in range(len(y_true)):
        abserr[i] = np.abs( (y_true[i] - y_pred[i]) / y_true[i])
    return 100 * np.mean(abserr)


# We run for every Regressor.
for Reg in RegDict:

    print("For : ",Reg)
    Model = RegDict[Reg]
    prog = 0
    y_GT = []
    y_PT = []
    abserror=[]

    # For every split obtained by Leave_One_Out split function.
    for i,(train_index, test_index) in enumerate(loo.split(X)):



        #m To show some sort of progress.
        prog = prog + 1
        print("Progress : {0}/{1}\r".format(prog,tot),end=" ")

        # Test-train split for the fold.
        X_Train, X_Test = X.iloc[train_index],X.iloc[test_index]
        y_Train, y_Test = y.iloc[train_index],y.iloc[test_index]
        
        Model.fit(X_Train, np.ravel(y_Train))
        pred = Model.predict(X_Test)  

        y_GT.append(y_Test.iloc[0,0])
        y_PT.append(pred[0]) 
 
        abserror.append(np.abs( (y_Test.iloc[0,0] - pred[0])/ y_Test.iloc[0,0] ))


    # Calculating Metrics
    MAPE = 100 * np.mean(abserror)
    # print("Calculated MAPE : ", MAPE )
    # print("Calculated MAPE_func : ", MAPE_func(y_GT,y_PT))
    print(" sklearm MAPE : " , 100 * mean_absolute_percentage_error(y_GT,y_PT))
    print(" sklearm MAE : " , mean_absolute_error(y_GT,y_PT))
    print(" sklearm MSE : " , mean_squared_error(y_GT,y_PT))





    