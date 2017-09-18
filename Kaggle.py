import pandas as pd
import numpy as np
import sys
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest

def clean_data(train_data, test_data):
    # Analyze data

    # Non-numeric by vision of box_plot in R

    # ------ EXAMPLE BOXPLOT IN PYTHON -------
    #
    # def msz_data(train_data):
    #     msz_dt = train_data[['MSZoning', 'SalePrice']]
    #     msz_dt.boxplot(['SalePrice'], 'MSZoning')
    #     plt.show()

    #Dropping columns

    clas_el = ['BedroomAbvGr', 'BldgType', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'HalfBath','LandSlope', 'LotConfig', 'MoSold', 'Utilities', 'YrSold']
    train = train_data.drop(clas_el, axis = 1)
    test = test_data.drop(clas_el, axis = 1)


    # Numeric by estimation of variance

    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

    num_data = train.select_dtypes(exclude=['object'])
    data_train = num_data.drop(['Id','SalePrice'],axis=1)


    #Drop Numeric Columns that are categorical and that I have already examined with boxplot

    numeric = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'EnclosedPorch', 'GarageArea', 'GarageYrBlt', 'GrLivArea', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF', '1stFlrSF', '2ndFlrSF', 'YearBuilt', 'YearRemodAdd' ]

    n_data = data_train[numeric].fillna(0)

    X = sel.fit_transform(n_data, y = train[['SalePrice']])

    #The numeric columns don't have a low variance, so I will keep
    #print(n_data.shape)
    #print(X.shape)


    #Outliers ---------------------------------------------------------------

    #Test and Train without NaN in the numeric culumns and without the dropped columns

    tr = data_train[numeric].fillna(0)
    ts = test[numeric].fillna(0)



    yTrain = train[['SalePrice']]


    clf = IsolationForest(n_estimators=500, max_samples=1.0, random_state=1001, bootstrap=True, contamination=0.02,
                          verbose=0, n_jobs=-1)
    clf.fit(tr.values, yTrain)
    isof = clf.predict(tr.values)
    tr.insert(0, 'SalePrice', yTrain)
    tr.insert(0, 'ID', train_data[['Id']])
    tr['isof'] = isof
    myindx = tr['isof'] < 0
    train_IF = tr.loc[myindx]
    train_IF.reset_index(drop=True, inplace=True)
    train_IF.to_csv('train-isof-outliers.csv', index=False)

    train['isof'] = tr['isof']
    c = train[train['isof'] >= 0]
    clean_train = c.drop(c[['isof']],axis = 1)

    ts['isof'] = clf.predict(ts.values)
    ts.insert(0, 'ID', test_data['Id'])
    myindex = ts['isof'] < 0
    test_IF = ts.loc[myindex]
    test_IF.reset_index(drop=True, inplace=True)
    test_IF.to_csv('test-isof-outliers.csv', index=False)


    test['isof'] = ts['isof']
    c = test[test['isof'] >= 0]
    clean_test = c.drop(c[['isof']],axis = 1)


    return(clean_train, clean_test)

def model(train, test):
    ytrain = train[['SalePrice']]
    xtrain =train.drop(ytrain, axis = 1)

    num = xtrain.select_dtypes(exclude=['object'])

    numeric = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'EnclosedPorch', 'GarageArea', 'GarageYrBlt', 'GrLivArea', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF', '1stFlrSF', '2ndFlrSF', 'YearBuilt', 'YearRemodAdd']

    xtrain_num = num[numeric]

    numeric = numeric + ['Id']

    xtrain_cat_num = num.drop(numeric, axis = 1)

    xtrain_cat_string = xtrain.select_dtypes(include = ['object'])

    xtrain_cat = pd.concat([xtrain_cat_num, xtrain_cat_string], axis = 1)

    xtrain_dummies = pd.get_dummies(xtrain_cat, dummy_na = True)

    print(xtrain_dummies.head(5))

    xtrain_all = pd.concat([xtrain_dummies, xtrain_num], axis = 1)
    print("Start training...")
    train_data = lgb.Dataset(xtrain_all.values, ytrain.values)
    test_data = lgb.Dataset(test.values, reference = train_data)

    # setting parameters for lightgbm
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1
    }
    # Here we have set max_depth in xgb and LightGBM to 7 to have a fair comparison between the two.
    # training our model using light gbm


    lgbm = lgb.train(params, train_data, num_boost_round = 20, early_stopping_rounds=5)

    lgbm.save_model('model.txt')

    lgb.cv(params, train_data, nfold=10)

    # predicting on test set
    ypred = lgbm.predict(test.values, num_iteration=lgbm.best_iteration)
    print(ypred[0:5])  # showing first 5 predictions


def main():
    print(sys.argv)
    filename = sys.argv[1]
    train_data = pd.read_csv(filename, sep = ',')
    filename2 = sys.argv[2]
    test_data = pd.read_csv(filename2, sep=',')


    train, test = clean_data(train_data, test_data)

    model(train, test)

if __name__ == "__main__":
    main()