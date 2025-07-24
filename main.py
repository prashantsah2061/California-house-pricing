import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import  SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score



MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy= 'median')),
        ('scaler', StandardScaler())

    ])

    cat_pipeline = Pipeline([
        ('encode', OneHotEncoder(handle_unknown= 'ignore'))
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):
    # 1. Load the dataset
    housing = pd.read_csv('/Users/prashantsah/Desktop/scikit-learn/california house pricing/housing.csv')

    #2. creating a stratified split
    housing['income_cat'] = pd.cut(housing['median_income'], bins = [0, 1.5, 3, 4.5, 6, np.inf], labels= [1,2,3,4,5])


    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state= 42)
    
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)
        strat_test_set.to_csv('input.csv', index = False)
        strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
        

    #4. seperating features and labels
    housing = strat_train_set.copy()
    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis=1)



    #5. Seperating numerical and categorical value
    nums_attribute = housing_features.drop('ocean_proximity', axis = 1).columns.tolist()
    cats_attribute = ['ocean_proximity']

    pipeline = build_pipeline(nums_attribute, cats_attribute)
    # print(housing_features)
    housing_prepared = pipeline.fit_transform(housing_features)
    

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model ,MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print('model is trained')

else:
    # lets do inference

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')

    transformed_input = pipeline.transform(input_data)
    prediction = model.predict(transformed_input)
    input_data['median_house_value'] = prediction

    input_data.to_csv('output.csv', index=False)
    print('result saved!!!!,  output.csv')