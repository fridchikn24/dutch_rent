from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
import pickle as pk
from model.pipeline.preparation import prepare_data
from config.config import settings


def build_model():

    df = prepare_data()

    X,y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    knn = train_knn(X_train,y_train)
    rf = train_rf(X_train,y_train)
    lgbm = train_lgbm(X_train, y_train)
    stacked_model = train_stack(X_train,y_train,knn,rf,lgbm)
    evaluate_model(stacked_model,X_test,y_test)
    save_model(stacked_model)

def get_X_y(data, 
            col_X = ['area', 
                  'constraction_year', 
                  'bedrooms', 'bathrooms',
                  'garden', 
                  'balcony_yes', 
                  'parking_yes', 
                  'furnished_yes', 
                  'garage_yes', 
                  'storage_yes'],
            col_y = 'rent'):

    return data[col_X], data[col_y]

def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2)
    
    return X_train, X_test, y_train, y_test

def train_knn(X_train,y_train):
    knn_params = {
    'n_neighbors': range(1, 20)
    }

    knn_grid = GridSearchCV(KNeighborsRegressor(), knn_params, cv=5, scoring='neg_root_mean_squared_error')
    knn_grid.fit(X_train, y_train)
    return knn_grid.best_estimator_

def train_rf(X_train,y_train):
    grid_space = {'n_estimators': [100, 200, 300], 'max_depth': [3, 6, 9, 12]}
    grid = GridSearchCV(RandomForestRegressor(), param_grid=grid_space, cv=5, scoring = 'neg_root_mean_squared_error')
    model_grid = grid.fit(X_train, y_train)
    return model_grid.best_estimator_

def train_lgbm(X_train,y_train):
    param_grid = {
    'num_leaves': [15, 31, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    }

    lgbm_grid = GridSearchCV(LGBMRegressor(), param_grid, cv=5, scoring='neg_root_mean_squared_error')
    lgbm_grid.fit(X_train, y_train)
    return lgbm_grid.best_estimator_

def train_stack(X_train,y_train,knn,rf,lgbm):
    stack = StackingRegressor(
    estimators=[('knn', knn), ('rf', rf), ('lgbm', lgbm)],
    final_estimator=RandomForestRegressor()
    )   

    stack_param = {
    'final_estimator__n_estimators': [100, 200, 300],
    'final_estimator__max_features':['sqrt','log2'],
    'final_estimator__max_depth': range(2,20,2)
    }

    stack_grid = GridSearchCV(stack, stack_param, cv=5, scoring='neg_root_mean_squared_error')

    stack_grid.fit(X_train, y_train)

    return stack_grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    return model.score(X_test, y_test)

def save_model(model):
    pk.dump(model,open(f'{settings.model_path}/{settings.model_name}','wb'))