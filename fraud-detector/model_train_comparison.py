######################################################################
# Model training pipeline for fraud detection, key highlights:
# - Different candidate models will be trained and assessed for this:
#  > logistic regression
#  > gradient boosting
#  > random forest
#  > catboost? (planned for future release)
# - The models will be built offline and locally, to take advantage of the sklearn library.
# - Pipeline of feature engineering, data splitting, preproccessing is applied before fitting.
# - Model fitting and hyperparameter optimization with randomised search.
# - Models are trained on increasing number of features, for variable selection purposes


# Other considerations:
# - Modelling functionality in Spark is steadily expanding, hence might move to PySpark in the future if more processing was required.
#   See https://spark.apache.org/docs/latest/ml-guide.html
# - Due to constrained timelines of this exercise, models were fitted on a random 10% sample as per configuration.

# TODO
# Look at feature importance and prioritise ordering used for variable selection
# Consider adding more types of classifiers such as catboost
# Update features based on any changes to eda + consider enlarging preprocessing e.g. bucketizer
# Bring preprocessing step forward so it is not repeated in various models and reduce runtime


######################################################################

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Binarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from feateng import FeatPipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from conf.modelling import sample_rate, inpath


###############################
# Variable Assignment and data read in
###############################
raw = load_data_local(inpath, intype).drop(columns = ["isFlaggedFraud"])
raw_sampled = pd.DataFrame.sample(raw, frac = sample_rate)

###############################
# Feature engineering step
###############################
df = FeatPipeline.fit_transform(raw_sampled)
print("Debugging log - FE step complete:", df.head())

###############################
# More variable Assignment
###############################
from conf.modelling import target_feature, id_features, numeric_features, categorical_features, binary_features
all_features = df.columns.difference(target_feature).difference(id_features)
other_features = all_features.difference(numeric_features).difference(categorical_features).difference(binary_features)

###############################
# Data splitting
###############################
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(df[all_features], np.array(df[target_feature]), test_size = 0.3,random_state=2, stratify=y)
print("Debugging log - Data splitting complete:", pd.DataFrame(X_train).head(), pd.DataFrame(y_train).head())

###############################
# Preprocessing pipeline
###############################
class ConvertToString(BaseEstimator, TransformerMixin):

       def __init__(self):
              pass

       def fit(self, X, y=None):
              return self

       def transform(self, X):
              return X.astype(str)

binary_transformer = Pipeline(steps = [
    ('imputer_bin', SimpleImputer(strategy = 'constant', fill_value = 0)),
    ('binarizer', Binarizer())
])
cat_transformer = Pipeline(steps = [
    #('Stringerise', ConvertToString()), ## Remove as did not export well in .pkl (#TODO fix)
    ('imputer_cat', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
    ('onehot', OneHotEncoder(sparse = False, handle_unknown='ignore'))
])
numeric_transformer = Pipeline(steps = [
    ('imputer_num', SimpleImputer(strategy = 'constant', fill_value = 0)),
    ('scaler', StandardScaler())
])


###############################
# Search space for hyperparameter optimization
# Volume to be searched where each dimension represents a hyperparameter and each point represents one model configuration.
###############################
# Commenting out some of the below to reduce runtime, can be augmented
lrSearchSpace = {
    'classifier': [LogisticRegression(n_jobs = -1, solver = 'saga')],
    'classifier__penalty': ['l2'],#, 'l1'],
    'classifier__class_weight': [{1: (i + 1) / 2} for i in range(2)],
    'classifier__C': [0.001, 1, 100]
#     ,'classifier__max_iter': range(100, 300)
}

rfSearchSpace = {
    'classifier': [RandomForestClassifier(n_jobs = -1)],
    'classifier__criterion': ['gini'], # , 'entropy'],
    'classifier__class_weight': [{1: (i + 1) / 2} for i in range(2)],
    #'classifier__n_estimators': range(100, 300),
    'classifier__max_depth': [3, 9, 15],
    'classifier__max_features': ["auto"]#, "sqrt", "log2"]
}

gbSearchSpace = {
    'classifier': [GradientBoostingClassifier()],
    'classifier__loss': ['deviance'], #, 'exponential'],
    #'classifier__n_estimators': range(100, 300),
    'classifier__max_depth': [3, 9, 15],
    'classifier__max_features': ["auto"]#, "sqrt", "log2"]
}

###############################
# Model fitting and hyperparameter optimization with randomised search.
# Also models are training on increasing number of features, for variable selection purposes
###############################
print("Training models on a increasing number of features. Total features:", len(all_features))

for i in range(len(all_features)):
       best_models = {}

       features = all_features[:i + 1]

       print(str(len(features)), " features: ", features)

       preprocessing = ColumnTransformer(transformers=[
              ('bin', binary_transformer, [X_train.columns.get_loc(x) for x in features if x in binary_features]),
              ('cat', cat_transformer, [X_train.columns.get_loc(x) for x in features if x in categorical_features]),
              ('num', numeric_transformer, [X_train.columns.get_loc(x) for x in features if x in numeric_features])
       ])

       lr_trainingPipeline = Pipeline(steps=[
              ('preprocessing', preprocessing),
              ('classifier', LogisticRegression())
       ])
       rf_trainingPipeline = Pipeline(steps=[
              ('preprocessing', preprocessing),
              ('classifier', RandomForestClassifier())
       ])
       gb_trainingPipeline = Pipeline(steps=[
              ('preprocessing', preprocessing),
              ('classifier', GradientBoostingClassifier())
       ])

       lr_search = RandomizedSearchCV(lr_trainingPipeline, param_distributions=lrSearchSpace, cv=2)
       rf_search = RandomizedSearchCV(rf_trainingPipeline, param_distributions=rfSearchSpace, cv=2)
       gb_search = RandomizedSearchCV(gb_trainingPipeline, param_distributions=gbSearchSpace, cv=2)

       print("lr_search"), lr_search.fit(X_train, y_train)
       print("rf_search"), rf_search.fit(X_train, y_train)
       print("gb_search"), gb_search.fit(X_train, y_train)

       best_models = {
              'LogisticRegression': lr_search.best_params_,
              'RandomForest': rf_search.best_params_,
              'GradientBoosting': gb_search.best_params_
       }

       joblib.dump(lr_trainingPipeline.set_params(**best_models['LogisticRegression']).fit(X_train, y_train),
                   f'../models/logisticRegression_{len(features)}.pkl');
       joblib.dump(rf_trainingPipeline.set_params(**best_models['RandomForest']).fit(X_train, y_train),
                   f'../models/randomForest_{len(features)}.pkl');
       joblib.dump(gb_trainingPipeline.set_params(**best_models['GradientBoosting']).fit(X_train, y_train),
                   f'../models/gradientBoosting_{len(features)}.pkl');

       print(f"{len(features)} done")

