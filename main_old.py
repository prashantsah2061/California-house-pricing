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

# 1. Load the dataset
housing = pd.read_csv('/Users/prashantsah/Desktop/scikit-learn/california house pricing/housing.csv')

#2. creating a stratified split
housing['income_cat'] = pd.cut(housing['median_income'], bins = [0, 1.5, 3, 4.5, 6, np.inf], labels= [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state= 42)
 
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis = 1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis = 1)


#3.  working on training data

housing  = strat_train_set.copy()

#4. seperating features and labels

housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis = 1)


#5. Seperating numerical and categorical value
nums_attribute = housing.drop('ocean_proximity', axis = 1).columns.tolist()
cats_attribute = ['ocean_proximity']


# 6. making pipeline for cateforical value
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy= "median")),
    ("scaler", StandardScaler())
])

# 7. making pipeline for numerical value
cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown= 'ignore'))
])

# 8. making full Pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, nums_attribute ),
    ('cat', cat_pipeline, cats_attribute)
])

# 9. transform the data

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# 10. LinearRegression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_root_mean_squared_error',  cv=10)
# print(f'The root mean squared error for linear regression is {lin_rmses}')
print(pd.Series(lin_rmses).describe())




# 11. Decision Tree model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring='neg_root_mean_squared_error',  cv=10)
# print(f'The root mean squared error for Decision Tree model is {dec_rmses}')
print(pd.Series(dec_rmses).describe())


# 12. Random Forest Tree model
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds = random_forest_reg.predict(housing_prepared)
# random_forest_rmse = root_mean_squared_error(housing_labels, random_forest_preds)
random_forest_rmses = -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring='neg_root_mean_squared_error',  cv=10)
# print(f'The root mean squared error for Random Forest Tree model is {random_forest_rmse}')
print(pd.Series(random_forest_rmses).describe())
