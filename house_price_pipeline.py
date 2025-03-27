import pandas as pd
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data_for_model=pickle.load(open("data_for_model.p","rb"))

X=data_for_model.drop("price_in_Lac",axis=1)
y=data_for_model["price_in_Lac"]

# Split train and test data set
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

#specific numeric and categorical features

categorical_features=["Address", "areaWithType", "transaction", "status", "furnishing"]
numeric_features=["square_feet", "price_per_sqft", "BHK"]

# Set-up pipeline
# Numerical features transformer

numeric_transformer=Pipeline(steps=[("imputer", SimpleImputer()),
                                    ("scaler", StandardScaler())])
categorical_transformer=Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="U")),
                                        ("ohe", OneHotEncoder(handle_unknown="ignore"))])

#Preprocessing pipeline

preprocessing_pipeline=ColumnTransformer(transformers= [("numeric", numeric_transformer, numeric_features),
                                                       ("categorical", categorical_transformer, categorical_features)])

# Applying the pipeline
#Linear regression

regressor= Pipeline(steps=[("preprocessing_pipeline", preprocessing_pipeline),
                    ("regressor", LinearRegression())])
regressor.fit(X_train, y_train)

y_pred_class=regressor.predict(X_test)
r2_score(y_test, y_pred_class)

# save the pipeline

import joblib
joblib.dump(regressor, "house_price_model.joblib")








