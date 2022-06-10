import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
#from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier

import warnings
warnings.filterwarnings("ignore")

seed = 7
np.random.seed(seed)

df = pd.read_csv("stock_data_technical.csv")

#Drop symbols with empty train or test data
df = df.loc[df.Symbol != "UNIPHOS"]
df = df.loc[df.Symbol != "TATATEA"]
df = df.loc[df.Symbol != "TATACONSUM"]
df = df.loc[df.Symbol != "MUNDRAPORT"]
df = df.loc[df.Symbol != "INFOSYSTCH"]
df = df.loc[df.Symbol != "HEROHONDA"]
df = df.loc[df.Symbol != "BAJAUTOFIN"]

#Drop symbols with imbalanced train or test data
df = df.loc[df.Symbol != "HDFCLIFE"]
df = df.loc[df.Symbol != "SBILIFE"]
df = df.loc[df.Symbol != "TATAGLOBAL"]

df.corr(method ='pearson').style.background_gradient(cmap='coolwarm').set_precision(3)

X = df[["Prev Close", "Open", "High", "Low","Last","Close","VWAP", "Volume", "Turnover", "Deliverable Volume", "%Deliverble"]]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

df = df[['Date', 'Symbol', 'Open', 'Volume', 'Turnover', 'Deliverable Volume','%Deliverble']]
df['Date']= pd.to_datetime(df['Date'])
df = df.sort_values(by = ["Symbol","Date"], ascending= False)

df["Label_3D"] = np.where(df["Open"].shift(3) >= df["Open"], 1, 0)
df["Label_7D"] = np.where(df["Open"].shift(7) >= df["Open"], 1, 0)
df["Label_30D"] = np.where(df["Open"].shift(30) >= df["Open"], 1, 0)
df["Label_90D"] = np.where(df["Open"].shift(90) >= df["Open"], 1, 0)

df["Symbol"] = df["Symbol"].astype('category')
Stocks = pd.concat([df["Symbol"], df["Symbol"].cat.codes], axis=1).drop_duplicates()
Stocks = Stocks.rename(columns={'Symbol':'cat_code'})
Stocks = Stocks.rename(columns={0:'Symbol'})
df["Symbol"] = df["Symbol"].cat.codes

df["Year"] = pd.DatetimeIndex(df['Date']).year
df["Month"] = pd.DatetimeIndex(df['Date']).month
df["day_of_week"] = df['Date'].apply(lambda x: x.weekday())

#LogScale 
df['Open'] = np.log(df['Open'])
df['Volume'] = np.log(df['Volume'])
df['Turnover'] = np.log(df['Turnover'])

df_train = df[df["Year"] < 2020]
df_test = df[df["Year"] > 2019]

#Check Symbolwise data size
LabelCountCheckTrain = df_train.groupby('Symbol').agg({'Symbol':'count', 'Label_3D': 'sum', 'Label_7D': 'sum', 'Label_30D': 'sum', 'Label_90D': 'sum'})
LabelCountCheckTrain = LabelCountCheckTrain.rename(columns={'Symbol':'Count_Symbols'}).reset_index()
LabelCountCheckTrain['Label_3D'] = LabelCountCheckTrain['Label_3D']/LabelCountCheckTrain['Count_Symbols']
LabelCountCheckTrain['Label_7D'] = LabelCountCheckTrain['Label_7D']/LabelCountCheckTrain['Count_Symbols']
LabelCountCheckTrain['Label_30D'] = LabelCountCheckTrain['Label_30D']/LabelCountCheckTrain['Count_Symbols']
LabelCountCheckTrain['Label_90D'] = LabelCountCheckTrain['Label_90D']/LabelCountCheckTrain['Count_Symbols']
LabelCountCheckTrain = pd.merge(LabelCountCheckTrain, Stocks, on ='Symbol', how ='inner')

LabelCountCheckTest = df_test.groupby('Symbol').agg({'Symbol':'count', 'Label_3D': 'sum', 'Label_7D': 'sum', 'Label_30D': 'sum', 'Label_90D': 'sum'})
LabelCountCheckTest = LabelCountCheckTest.rename(columns={'Symbol':'Count_Symbols'}).reset_index()
LabelCountCheckTest['Label_3D'] = LabelCountCheckTest['Label_3D']/LabelCountCheckTest['Count_Symbols']
LabelCountCheckTest['Label_7D'] = LabelCountCheckTest['Label_7D']/LabelCountCheckTest['Count_Symbols']
LabelCountCheckTest['Label_30D'] = LabelCountCheckTest['Label_30D']/LabelCountCheckTest['Count_Symbols']
LabelCountCheckTest['Label_90D'] = LabelCountCheckTest['Label_90D']/LabelCountCheckTest['Count_Symbols']
LabelCountCheckTest = pd.merge(LabelCountCheckTest, Stocks, on ='Symbol', how ='inner')

X = df[['Open', 'Volume', 'Turnover', '%Deliverble', 'Label_3D', 'Label_7D', 'Label_30D', 'Label_90D', 'Year', 'Month']]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

X_Col = ['Open','Symbol', '%Deliverble', 'Year','Month', 'day_of_week']
Y_Col = ['Label_3D', 'Label_7D', 'Label_30D', 'Label_90D']

Score = pd.DataFrame(columns=['Symbol','PredRange','Years_Train','Method','Train_Accuracy','Train_F1','Train_Precision','Train_Recall','Test_Accuracy','Test_F1','Test_Precision','Test_Recall','CorrTrainMax','CorrTestMax'])

for i in Y_Col:

    print(i)
    X_train = df_train[X_Col]
    X_test = df_test[X_Col]
    Corr_Train = X_train.corr(method ='pearson')
    Corr_Test = X_test.corr(method ='pearson')

    Y_train = df_train[i]
    Y_test = df_test[i]        

    print(i,"LR")    
    log_LR = LogisticRegression()
    param_grid={"C":np.logspace(-10,10,10)}
    grid_search_LR = GridSearchCV(log_LR,param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
    grid_search_LR.fit(X_train, Y_train)    
    bestLRParams = grid_search_LR.best_params_
    log_LR = LogisticRegression(C=bestLRParams.get('C')).fit(X_train, Y_train) 
    Y_Pred_Train = log_LR.predict(X_train)
    Y_Pred_Test = log_LR.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "Logistic", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    
    print(i,"NB")        
    Naive_Bayes = GaussianNB()
    param_grid={"var_smoothing":np.logspace(0,-9,10)}
    Naive_Bayes.fit(X_train, Y_train)
    grid_search_NB = GridSearchCV(Naive_Bayes,param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
    grid_search_NB.fit(X_train, Y_train)
    bestNBParams = grid_search_NB.best_params_
    Naive_Bayes = GaussianNB(var_smoothing=bestNBParams.get('var_smoothing')).fit(X_train, Y_train)
    Y_Pred_Train = Naive_Bayes.predict(X_train)
    Y_Pred_Test = Naive_Bayes.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "Naive_Bayes", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    
    '''
    param_grid = {'C': [0.01, 1, 100], 'gamma': [1, 0.1, 0.001], 'kernel': ['linear','rbf']} 
    svmc = svm.SVC()
    grid_search_SVM = GridSearchCV(svmc, param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
    grid_search_SVM.fit(X_train, Y_train)
    bestSVMParams = grid_search_SVM.best_params_
    svmc = svm.SVC(C = bestSVMParams.get('C'), gamma = bestSVMParams.get('gamma'), kernel=bestSVMParams.get('kernel')).fit(X_train, Y_train) 
    Y_Pred_Train = svmc.predict(X_train)
    Y_Pred_Test = svmc.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "SVM", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    '''
    print(i,"KNN")    
    knn = KNeighborsClassifier()
    k_range = list(range(2, 42, 4))
    param_grid = dict(n_neighbors = k_range)
    grid_search_KNN = GridSearchCV(knn, param_grid=param_grid, cv=5, return_train_score=False, scoring='precision', verbose = True, n_jobs=-1)
    grid_search_KNN.fit(X_train, Y_train)
    bestKNNParams = grid_search_KNN.best_params_
    knn = KNeighborsClassifier(n_neighbors = bestKNNParams.get('n_neighbors')).fit(X_train, Y_train) 
    Y_Pred_Train = knn.predict(X_train)
    Y_Pred_Test = knn.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "KNN", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    
    print(i,"DTC")    
    param_grid = {'max_leaf_nodes': list(range(2, 50, 2)), 'criterion' :['gini', 'entropy']}
    dtc = tree.DecisionTreeClassifier()
    grid_search_dtc = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5,  scoring='precision', verbose = True, n_jobs=-1)
    grid_search_dtc.fit(X_train, Y_train)
    bestDTCParams = grid_search_dtc.best_params_
    dtc = tree.DecisionTreeClassifier(criterion = bestDTCParams.get('criterion'), max_leaf_nodes = bestDTCParams.get('max_leaf_nodes')).fit(X_train, Y_train) 
    Y_Pred_Train = dtc.predict(X_train)
    Y_Pred_Test = dtc.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "Decision_Tree", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    
    print(i,"RF")    
    param_grid = [{'max_depth': list(range(2,7)), 'n_estimators': list(range(60,100,10))}]
    rf = RandomForestClassifier()    
    grid_search_rf = GridSearchCV(rf, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
    grid_search_rf.fit(X_train, Y_train)
    bestRFCParams = grid_search_rf.best_params_
    rf = RandomForestClassifier(max_depth = bestRFCParams.get('max_depth'), n_estimators = bestRFCParams.get('n_estimators')).fit(X_train, Y_train) 
    Y_Pred_Train = rf.predict(X_train)
    Y_Pred_Test = rf.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "Random_Forest", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    
    print(i,"LDA")    
    param_grid = {'solver': ['svd', 'lsqr', 'eigen']}
    lda = LDA()
    grid_search_lda = GridSearchCV(lda, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
    grid_search_lda.fit(X_train, Y_train)
    bestLDAParams = grid_search_lda.best_params_
    lda = LDA(solver = bestLDAParams.get('solver')).fit(X_train, Y_train)    
    Y_Pred_Train = lda.predict(X_train)
    Y_Pred_Test = lda.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "LDA", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)

    print(i,"QDA")    
    param_grid = {'reg_param': (0.000001,0.00001,0.0001,0.001), 'tol': (0.000001,0.00001, 0.0001, 0.001)}
    qda = QDA()
    grid_search_qda = GridSearchCV(qda, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
    grid_search_qda.fit(X_train, Y_train)
    bestQDAParams = grid_search_qda.best_params_
    qda = QDA(reg_param = bestQDAParams.get('reg_param'), tol = bestQDAParams.get('tol')).fit(X_train, Y_train)    
    Y_Pred_Train = qda.predict(X_train)
    Y_Pred_Test = qda.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "QDA", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)   
    
    for s in range(Stocks.shape[0]):
        cat_code = Stocks.iloc[s,1]
        
        X_train = df_train.loc[df_train["Symbol"] == cat_code][X_Col]
        X_test = df_test.loc[df_test["Symbol"] == cat_code][X_Col]

        Y_train = df_train.loc[df_train["Symbol"] == cat_code][i]
        Y_test = df_test.loc[df_test["Symbol"] == cat_code][i]

        print(i,Stocks.iloc[s,0],"LR")
        log_LR = LogisticRegression()
        param_grid={"C":np.logspace(-10,10,10)}
        grid_search = GridSearchCV(log_LR,param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)    
        bestLRParams = grid_search.best_params_
        log_LR = LogisticRegression(C=bestLRParams.get('C')).fit(X_train, Y_train) 
        Y_Pred_Train = log_LR.predict(X_train)
        Y_Pred_Test = log_LR.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "Logistic", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
        
        print(i,Stocks.iloc[s,0],"NB")
        Naive_Bayes = GaussianNB()
        param_grid={"var_smoothing":np.logspace(0,-9,10)}
        Naive_Bayes.fit(X_train, Y_train)
        grid_search = GridSearchCV(Naive_Bayes,param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestNBParams = grid_search.best_params_
        Naive_Bayes = GaussianNB(var_smoothing=bestNBParams.get('var_smoothing')).fit(X_train, Y_train)
        Y_Pred_Train = Naive_Bayes.predict(X_train)
        Y_Pred_Test = Naive_Bayes.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "Naive_Bayes", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
        
        '''
        param_grid = {'C': [0.01, 1, 100], 'gamma': [1, 0.1, 0.001], 'kernel': ['linear','rbf']} 
        svmc = svm.SVC()
        grid_search = GridSearchCV(svmc, param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestSVMParams = grid_search.best_params_
        svmc = svm.SVC(C = bestSVMParams.get('C'), gamma = bestSVMParams.get('gamma'), kernel=bestSVMParams.get('kernel')).fit(X_train, Y_train) 
        Y_Pred_Train = svmc.predict(X_train)
        Y_Pred_Test = svmc.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "SVM", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
        '''
        
        print(i,Stocks.iloc[s,0],"KNN")
        knn = KNeighborsClassifier()
        k_range = list(range(2, 42, 4))
        param_grid = dict(n_neighbors = k_range)
        grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5, return_train_score=False, scoring='precision', verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestKNNParams = grid_search.best_params_
        knn = KNeighborsClassifier(n_neighbors = bestKNNParams.get('n_neighbors')).fit(X_train, Y_train) 
        Y_Pred_Train = knn.predict(X_train)
        Y_Pred_Test = knn.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "KNN", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)

        print(i,Stocks.iloc[s,0],"DTC")  
        param_grid = {'max_leaf_nodes': list(range(2, 50, 2)), 'criterion' :['gini', 'entropy']}
        dtc = tree.DecisionTreeClassifier()
        grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5,  scoring='precision', verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestDTCParams = grid_search.best_params_
        dtc = tree.DecisionTreeClassifier(criterion = bestDTCParams.get('criterion'), max_leaf_nodes = bestDTCParams.get('max_leaf_nodes')).fit(X_train, Y_train) 
        Y_Pred_Train = dtc.predict(X_train)
        Y_Pred_Test = dtc.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "Decision_Tree", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)

        print(i,Stocks.iloc[s,0],"RF")   
        param_grid = [{'max_depth': list(range(2,7)), 'n_estimators': list(range(60,100,10))}]
        rf = RandomForestClassifier()    
        grid_search = GridSearchCV(rf, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestRFCParams = grid_search.best_params_
        rf = RandomForestClassifier(max_depth = bestRFCParams.get('max_depth'), n_estimators = bestRFCParams.get('n_estimators')).fit(X_train, Y_train) 
        Y_Pred_Train = rf.predict(X_train)
        Y_Pred_Test = rf.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "Random_Forest", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
            
        print(i,Stocks.iloc[s,0],"LDA")   
        param_grid = {'solver': ['svd', 'lsqr', 'eigen']}
        lda = LDA()
        grid_search = GridSearchCV(lda, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestLDAParams = grid_search.best_params_
        lda = LDA(solver = bestLDAParams.get('solver')).fit(X_train, Y_train)    
        Y_Pred_Train = lda.predict(X_train)
        Y_Pred_Test = lda.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "LDA", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)

        print(i,Stocks.iloc[s,0],"QDA")  
        param_grid = {'reg_param': (0.000001,0.00001,0.0001,0.001), 'tol': (0.000001,0.00001, 0.0001, 0.001)}
        qda = QDA()
        grid_search = GridSearchCV(qda, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestQDAParams = grid_search.best_params_
        qda = QDA(reg_param = bestQDAParams.get('reg_param'), tol = bestQDAParams.get('tol')).fit(X_train, Y_train)    
        Y_Pred_Train = qda.predict(X_train)
        Y_Pred_Test = qda.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "QDA", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)   

Score.to_csv("Tech_Pred_Results_with_gridsearch.csv",index = False)
############################################## Change Training data to more recent #########################################
df_train2 = df[df["Year"] < 2021]
df_train2 = df_train2[df_train2["Year"] > 2015]
df_test2 = df[df["Year"] > 2020]

#Check Symbolwise data size
LabelCountCheckTrain2 = df_train2.groupby('Symbol').agg({'Symbol':'count', 'Label_3D': 'sum', 'Label_7D': 'sum', 'Label_30D': 'sum', 'Label_90D': 'sum'})
LabelCountCheckTrain2 = LabelCountCheckTrain2.rename(columns={'Symbol':'Count_Symbols'}).reset_index()
LabelCountCheckTrain2['Label_3D'] = LabelCountCheckTrain2['Label_3D']/LabelCountCheckTrain2['Count_Symbols']
LabelCountCheckTrain2['Label_7D'] = LabelCountCheckTrain2['Label_7D']/LabelCountCheckTrain2['Count_Symbols']
LabelCountCheckTrain2['Label_30D'] = LabelCountCheckTrain2['Label_30D']/LabelCountCheckTrain2['Count_Symbols']
LabelCountCheckTrain2['Label_90D'] = LabelCountCheckTrain2['Label_90D']/LabelCountCheckTrain2['Count_Symbols']
LabelCountCheckTrain2 = pd.merge(LabelCountCheckTrain2, Stocks, on ='Symbol', how ='inner')

LabelCountCheckTest2 = df_test2.groupby('Symbol').agg({'Symbol':'count', 'Label_3D': 'sum', 'Label_7D': 'sum', 'Label_30D': 'sum', 'Label_90D': 'sum'})
LabelCountCheckTest2 = LabelCountCheckTest2.rename(columns={'Symbol':'Count_Symbols'}).reset_index()
LabelCountCheckTest2['Label_3D'] = LabelCountCheckTest2['Label_3D']/LabelCountCheckTest2['Count_Symbols']
LabelCountCheckTest2['Label_7D'] = LabelCountCheckTest2['Label_7D']/LabelCountCheckTest2['Count_Symbols']
LabelCountCheckTest2['Label_30D'] = LabelCountCheckTest2['Label_30D']/LabelCountCheckTest2['Count_Symbols']
LabelCountCheckTest2['Label_90D'] = LabelCountCheckTest2['Label_90D']/LabelCountCheckTest2['Count_Symbols']
LabelCountCheckTest2 = pd.merge(LabelCountCheckTest2, Stocks, on ='Symbol', how ='inner')

for i in Y_Col:
    print(i)

    X_train = df_train2[X_Col]
    X_test = df_test2[X_Col]

    Y_train = df_train2[i]
    Y_test = df_test2[i]        
    
    log_LR = LogisticRegression()
    param_grid={"C":np.logspace(-10,10,10)}
    grid_search_LR2 = GridSearchCV(log_LR,param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
    grid_search_LR2.fit(X_train, Y_train)    
    bestLRParams = grid_search_LR2.best_params_
    log_LR = LogisticRegression(C=bestLRParams.get('C')).fit(X_train, Y_train) 
    Y_Pred_Train = log_LR.predict(X_train)
    Y_Pred_Test = log_LR.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "Logistic", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    
    Naive_Bayes = GaussianNB()
    param_grid={"var_smoothing":np.logspace(0,-9,10)}
    Naive_Bayes.fit(X_train, Y_train)
    grid_search_NB2 = GridSearchCV(Naive_Bayes,param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
    grid_search_NB2.fit(X_train, Y_train)
    bestNBParams = grid_search_NB2.best_params_
    Naive_Bayes = GaussianNB(var_smoothing=bestNBParams.get('var_smoothing')).fit(X_train, Y_train)
    Y_Pred_Train = Naive_Bayes.predict(X_train)
    Y_Pred_Test = Naive_Bayes.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "Naive_Bayes", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    
    '''
    param_grid = {'C': [0.01, 1, 100], 'gamma': [1, 0.1, 0.001], 'kernel': ['linear','rbf']} 
    svmc = svm.SVC()
    grid_search_SVM2 = GridSearchCV(svmc, param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
    grid_search_SVM2.fit(X_train, Y_train)
    bestSVMParams = grid_search_SVM2.best_params_
    svmc = svm.SVC(C = bestSVMParams.get('C'), gamma = bestSVMParams.get('gamma'), kernel=bestSVMParams.get('kernel')).fit(X_train, Y_train) 
    Y_Pred_Train = svmc.predict(X_train)
    Y_Pred_Test = svmc.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "SVM", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    '''
    
    knn = KNeighborsClassifier()
    k_range = list(range(2, 42, 4))
    param_grid = dict(n_neighbors = k_range)
    grid_search_KNN2 = GridSearchCV(knn, param_grid=param_grid, cv=5, return_train_score=False, scoring='precision', verbose = True, n_jobs=-1)
    grid_search_KNN2.fit(X_train, Y_train)
    bestKNNParams = grid_search_KNN2.best_params_
    knn = KNeighborsClassifier(n_neighbors = bestKNNParams.get('n_neighbors')).fit(X_train, Y_train) 
    Y_Pred_Train = knn.predict(X_train)
    Y_Pred_Test = knn.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "KNN", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)

    param_grid = {'max_leaf_nodes': list(range(2, 50, 2)), 'criterion' :['gini', 'entropy']}
    dtc = tree.DecisionTreeClassifier()
    grid_search_dtc2 = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5,  scoring='precision', verbose = True, n_jobs=-1)
    grid_search_dtc2.fit(X_train, Y_train)
    bestDTCParams = grid_search_dtc2.best_params_
    dtc = tree.DecisionTreeClassifier(criterion = bestDTCParams.get('criterion'), max_leaf_nodes = bestDTCParams.get('max_leaf_nodes')).fit(X_train, Y_train) 
    Y_Pred_Train = dtc.predict(X_train)
    Y_Pred_Test = dtc.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "Decision_Tree", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)

    param_grid = [{'max_depth': list(range(2,7)), 'n_estimators': list(range(60,100,10))}]
    rf = RandomForestClassifier()    
    grid_search_rf2 = GridSearchCV(rf, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
    grid_search_rf2.fit(X_train, Y_train)
    bestRFCParams = grid_search_rf2.best_params_
    rf = RandomForestClassifier(max_depth = bestRFCParams.get('max_depth'), n_estimators = bestRFCParams.get('n_estimators')).fit(X_train, Y_train) 
    Y_Pred_Train = rf.predict(X_train)
    Y_Pred_Test = rf.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "Random_Forest", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    
    param_grid = {'n_components': [1,2], 'solver': ['svd', 'lsqr', 'eigen']}
    lda = LDA()
    grid_search_lda2 = GridSearchCV(lda, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
    grid_search_lda2.fit(X_train, Y_train)
    bestLDAParams = grid_search_lda2.best_params_
    lda = LDA(n_components = bestLDAParams.get('n_components'), solver = bestLDAParams.get('solver')).fit(X_train, Y_train)    
    Y_Pred_Train = lda.predict(X_train)
    Y_Pred_Test = lda.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "LDA", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)

    param_grid = {'reg_param': (0.000001,0.00001,0.0001,0.001), 'tol': (0.000001,0.00001, 0.0001, 0.001)}
    qda = QDA()
    grid_search_qda2 = GridSearchCV(qda, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
    grid_search_qda2.fit(X_train, Y_train)
    bestQDAParams = grid_search_qda2.best_params_
    qda = QDA(reg_param = bestQDAParams.get('reg_param'), tol = bestQDAParams.get('tol')).fit(X_train, Y_train)    
    Y_Pred_Train = qda.predict(X_train)
    Y_Pred_Test = qda.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "QDA", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)   
    
    for s in range(Stocks.shape[0]):
        print(Stocks.iloc[s,0])
        cat_code = Stocks.iloc[s,1]
        
        X_train = df_train2.loc[df_train2["Symbol"] == cat_code][X_Col]
        X_test = df_test2.loc[df_test2["Symbol"] == cat_code][X_Col]

        Y_train = df_train2.loc[df_train2["Symbol"] == cat_code][i]
        Y_test = df_test2.loc[df_test2["Symbol"] == cat_code][i]        
        
        log_LR = LogisticRegression()
        param_grid={"C":np.logspace(-10,10,10)}
        grid_search = GridSearchCV(log_LR,param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)    
        bestLRParams = grid_search.best_params_
        log_LR = LogisticRegression(C=bestLRParams.get('C')).fit(X_train, Y_train) 
        Y_Pred_Train = log_LR.predict(X_train)
        Y_Pred_Test = log_LR.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "Logistic", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
        
        Naive_Bayes = GaussianNB()
        param_grid={"var_smoothing":np.logspace(0,-9,10)}
        Naive_Bayes.fit(X_train, Y_train)
        grid_search = GridSearchCV(Naive_Bayes,param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestNBParams = grid_search.best_params_
        Naive_Bayes = GaussianNB(var_smoothing=bestNBParams.get('var_smoothing')).fit(X_train, Y_train)
        Y_Pred_Train = Naive_Bayes.predict(X_train)
        Y_Pred_Test = Naive_Bayes.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "Naive_Bayes", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
        
        '''
        param_grid = {'C': [0.01, 1, 100], 'gamma': [1, 0.1, 0.001], 'kernel': ['linear','rbf']} 
        svmc = svm.SVC()
        grid_search = GridSearchCV(svmc, param_grid=param_grid,cv=5, scoring='precision', verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestSVMParams = grid_search.best_params_
        svmc = svm.SVC(C = bestSVMParams.get('C'), gamma = bestSVMParams.get('gamma'), kernel=bestSVMParams.get('kernel')).fit(X_train, Y_train) 
        Y_Pred_Train = svmc.predict(X_train)
        Y_Pred_Test = svmc.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "SVM", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
        '''
        
        knn = KNeighborsClassifier()
        k_range = list(range(2, 42, 4))
        param_grid = dict(n_neighbors = k_range)
        grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5, return_train_score=False, scoring='precision', verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestKNNParams = grid_search.best_params_
        knn = KNeighborsClassifier(n_neighbors = bestKNNParams.get('n_neighbors')).fit(X_train, Y_train) 
        Y_Pred_Train = knn.predict(X_train)
        Y_Pred_Test = knn.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "KNN", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)

        param_grid = {'max_leaf_nodes': list(range(2, 50, 2)), 'criterion' :['gini', 'entropy']}
        dtc = tree.DecisionTreeClassifier()
        grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5,  scoring='precision', verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestDTCParams = grid_search.best_params_
        dtc = tree.DecisionTreeClassifier(criterion = bestDTCParams.get('criterion'), max_leaf_nodes = bestDTCParams.get('max_leaf_nodes')).fit(X_train, Y_train) 
        Y_Pred_Train = dtc.predict(X_train)
        Y_Pred_Test = dtc.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "Decision_Tree", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)

        param_grid = [{'max_depth': list(range(2,7)), 'n_estimators': list(range(60,100,10))}]
        rf = RandomForestClassifier()    
        grid_search = GridSearchCV(rf, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestRFCParams = grid_search.best_params_
        rf = RandomForestClassifier(max_depth = bestRFCParams.get('max_depth'), n_estimators = bestRFCParams.get('n_estimators')).fit(X_train, Y_train) 
        Y_Pred_Train = rf.predict(X_train)
        Y_Pred_Test = rf.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "Random_Forest", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
        
        param_grid = {'n_components': [1,2], 'solver': ['svd', 'lsqr', 'eigen']}
        lda = LDA()
        grid_search = GridSearchCV(lda, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestLDAParams = grid_search.best_params_
        lda = LDA(n_components = bestLDAParams.get('n_components'), solver = bestLDAParams.get('solver')).fit(X_train, Y_train)    
        Y_Pred_Train = lda.predict(X_train)
        Y_Pred_Test = lda.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "LDA", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)

        param_grid = {'reg_param': (0.000001,0.00001,0.0001,0.001), 'tol': (0.000001,0.00001, 0.0001, 0.001)}
        qda = QDA()
        grid_search = GridSearchCV(qda, param_grid=param_grid, cv = 5, scoring='precision',verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestQDAParams = grid_search.best_params_
        qda = QDA(reg_param = bestQDAParams.get('reg_param'), tol = bestQDAParams.get('tol')).fit(X_train, Y_train)    
        Y_Pred_Train = qda.predict(X_train)
        Y_Pred_Test = qda.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "QDA", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)   
        

Score.to_csv("Tech_Pred_Results_with_gridsearch.csv",index = False)

for i in Y_Col:

    X_train = df_train[X_Col]
    X_test = df_test[X_Col]

    Y_train = df_train[i]
    Y_test = df_test[i]        
    
    param_grid = {'C': [0.01, 1, 100], 'gamma': [1, 0.1, 0.001], 'kernel': ['linear','rbf']} 
    svmc = svm.SVC()
    grid_search_SVM = GridSearchCV(svmc, param_grid = param_grid, cv = 2, scoring='precision',verbose = True, n_jobs=-1)
    grid_search_SVM.fit(X_train, Y_train)
    bestSVMParams = grid_search_SVM.best_params_
    svmc = svm.SVC(C = bestSVMParams.get('C'), gamma = bestSVMParams.get('gamma'), kernel=bestSVMParams.get('kernel'))
    svmc.fit(X_train, Y_train)
    Y_Pred_Train = svmc.predict(X_train)
    Y_Pred_Test = svmc.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "SVM", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    '''    
    for s in range(Stocks.shape[0]):
        #print(Stocks.iloc[s,0])
        cat_code = Stocks.iloc[s,1]
        
        X_train = df_train.loc[df_train["Symbol"] == cat_code][X_Col]
        X_test = df_test.loc[df_test["Symbol"] == cat_code][X_Col]

        Y_train = df_train.loc[df_train["Symbol"] == cat_code][i]
        Y_test = df_test.loc[df_test["Symbol"] == cat_code][i]        
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        param_grid = {'C': [0.01, 1, 100], 'gamma': [1, 0.1, 0.001], 'kernel': ['linear','rbf']} 
        svmc = svm.SVC()
        grid_search = GridSearchCV(svmc,param_grid= param_grid, cv = 2, scoring='precision',verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestSVMParams = grid_search.best_params_
        svmc = svm.SVC(C = bestSVMParams.get('C'), gamma = bestSVMParams.get('gamma'), kernel=bestSVMParams.get('kernel'))
        svmc.fit(X_train, Y_train)
        Y_Pred_Train = svmc.predict(X_train)
        Y_Pred_Test = svmc.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2010-19", 'Method' : "SVM", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    '''
    X_train = df_train2[X_Col]
    X_test = df_test2[X_Col]

    Y_train = df_train2[i]
    Y_test = df_test2[i]        
    
    param_grid = {'C': [0.01, 1, 100], 'gamma': [1, 0.1, 0.001], 'kernel': ['linear','rbf']} 
    svmc = svm.SVC()
    grid_search_SVM2 = GridSearchCV(svmc,param_grid = param_grid, cv = 2, scoring='precision',verbose = True, n_jobs=-1)
    grid_search_SVM2.fit(X_train, Y_train)
    bestSVMParams = grid_search_SVM2.best_params_
    svmc = svm.SVC(C = bestSVMParams.get('C'), gamma = bestSVMParams.get('gamma'), kernel=bestSVMParams.get('kernel'))
    svmc.fit(X_train, Y_train)
    Y_Pred_Train = svmc.predict(X_train)
    Y_Pred_Test = svmc.predict(X_test)
    Score = Score.append({'Symbol' : "ALL", 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "SVM", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    '''
    for s in range(Stocks.shape[0]):
        #print(Stocks.iloc[s,0])
        cat_code = Stocks.iloc[s,1]
        
        X_train = df_train2.loc[df_train2["Symbol"] == cat_code][X_Col]
        X_test = df_test2.loc[df_test2["Symbol"] == cat_code][X_Col]
        
        Y_train = df_train2.loc[df_train2["Symbol"] == cat_code][i]
        Y_test = df_test2.loc[df_test2["Symbol"] == cat_code][i]        
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        param_grid = {'C': [0.01, 1, 100], 'gamma': [1, 0.1, 0.001], 'kernel': ['linear','rbf']} 
        svmc = svm.SVC()
        grid_search = GridSearchCV(svmc,param_grid = param_grid, cv = 2, scoring='precision',verbose = True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        bestSVMParams = grid_search.best_params_
        svmc = svm.SVC(C = bestSVMParams.get('C'), gamma = bestSVMParams.get('gamma'), kernel=bestSVMParams.get('kernel'))
        svmc.fit(X_train, Y_train)
        Y_Pred_Train = svmc.predict(X_train)
        Y_Pred_Test = svmc.predict(X_test)
        Score = Score.append({'Symbol' : Stocks.iloc[s,0], 'PredRange' : i, 'Years_Train' : "2016-20", 'Method' : "SVM", 'Train_Accuracy' : round(accuracy_score(Y_train, Y_Pred_Train),3), 'Train_F1' : round(f1_score(Y_train, Y_Pred_Train),3), 'Train_Precision' : round(precision_score(Y_train, Y_Pred_Train),3), 'Train_Recall' : round(recall_score(Y_train, Y_Pred_Train),3), 'Test_Accuracy' : round(accuracy_score(Y_test, Y_Pred_Test),3), 'Test_F1' : round(f1_score(Y_test, Y_Pred_Test),3), 'Test_Precision' : round(precision_score(Y_test, Y_Pred_Test),3), 'Test_Recall' : round(recall_score(Y_test, Y_Pred_Test),3)}, ignore_index = True)
    '''
    
Score.to_csv("Tech_Pred_Results_with_gridsearch.csv",index = False)
