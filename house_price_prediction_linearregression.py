#importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import RFECV

data=pd.read_excel("surat.xlsx")

# Data Cleaning

cat_cols=data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col]=data[col].str.strip()
    
#filter only Apartment type properties

data_for_model=data[data["Type"]=="Apartment"]

msno.bar(data_for_model)

data_for_model.drop(["Type", "facing", "description", "floor"], axis=1, inplace=True)
data_for_model.dropna(subset = ['Address', 'furnishing', 'transaction', 'status'], inplace=True)

data_for_model = data_for_model[~(data_for_model['price'] == 'Call for Price')]
data_for_model['price'] = data_for_model['price'].str.replace('?', '')
data_for_model["price_in_Lac"]=data_for_model['price'].replace({'Cr':'*1e2','Lac':"*1"}, regex=True).map(pd.eval).astype(float)

data_for_model.drop(["price"], axis=1, inplace=True)
data_for_model.dropna(subset=["square_feet"], inplace=True)
data_for_model["price_per_sqft"].fillna((data_for_model["price_in_Lac"]/data_for_model["square_feet"]), inplace=True)

# creating new column BHK
data_for_model["Property"].unique()

i=data_for_model[(data_for_model["Property"]=="Studio")].index
data_for_model.drop(i, inplace=True)

data_for_model["BHK"]=data_for_model["Property"].apply(lambda x: int(x.split(' ')[0]))
data_for_model.drop("Property", axis=1, inplace=True)

# Finding the sub-groups in categorical columns

cat_cols=data_for_model[["areaWithType", "transaction", "furnishing"]]
for col in cat_cols:
    print(data_for_model[col].value_counts())

number= data_for_model["status"].nunique()
print(f"number of unique status: {number}")

data_for_model.loc[data_for_model['status'] != "Ready to Move", 'status'] = "still in construction"
print(data_for_model["status"].value_counts())

address_numbers=len(data_for_model["Address"].unique())
print(f"Total Address before cleaning the data: {address_numbers}")

Address_stats=data_for_model.groupby("Address")["Address"].agg("count").sort_values(ascending=False)
print(f" Address counts less than 10: {len(Address_stats[Address_stats<=10])}")

Address_stats_less_than_10=Address_stats[Address_stats<=10]
data_for_model.Address=data_for_model.Address.apply(lambda x: 'other' if x in Address_stats_less_than_10 else x)
print(f"After grouping addresses number of unique address: {data_for_model.Address.nunique()}")

# Outliers

#1.Removing apartments which are having sqft less than 300 per BHK

print(f"Total number of apartments less than 300sqft per BHK: {len(data_for_model[data_for_model.square_feet/data_for_model.BHK<300])}")
data_for_model=data_for_model[~(data_for_model.square_feet/data_for_model.BHK<300)]

#2. Removing outliers in price_per_sqft column

data_for_model["price_per_sqft"].describe()

def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('Address'):
        m = np.mean(subdf.price_per_sqft)
        sd = np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-sd)) & (subdf.price_per_sqft<=(m+sd))]
        df_out=pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

data_for_model = remove_pps_outliers(data_for_model)

def plot_scatter_chart(df,Address):
    bhk2= df[(df.Address==Address) & (df.BHK==2)]
    bhk3= df[(df.Address==Address) & (df.BHK==3)]
    plt.scatter(bhk2.square_feet, bhk2.price_in_Lac, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.square_feet, bhk3.price_in_Lac, marker = '+', color='red', label='3 BHK', s=50)
    plt.xlabel("Square feet area")
    plt.ylabel("price_in_Lac")
    plt.title(Address)
    plt.legend()          

plot_scatter_chart(data_for_model, "Palanpur")

# Splitting input and output variable
X=data_for_model.drop(["price_in_Lac"], axis=1)
y=data_for_model["price_in_Lac"]

# Split train and test dataset
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

# Dealing with Categorical data
categorical_var = ['Address', "areaWithType", "transaction", "status", "furnishing"]
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_var])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_var])

encoder_feature_names= one_hot_encoder.get_feature_names_out(categorical_var)

X_train_encoded=pd.DataFrame(X_train_encoded, columns=encoder_feature_names)
X_train=pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_var, axis=1, inplace=True)

X_test_encoded=pd.DataFrame(X_test_encoded, columns=encoder_feature_names)
X_test=pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(categorical_var, axis=1, inplace=True)

#Feature scaling
scale_norm = StandardScaler()
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns = X_test.columns)

# Feature Selection
regressor=LinearRegression()
feature_selector = RFECV(regressor)

fit=feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of feature: {optimal_feature_count}")

X_train=X_train.loc[:, feature_selector.get_support()]
X_test=X_test.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.cv_results_['mean_test_score'])+1), fit.cv_results_['mean_test_score'], marker='o')  # grid starts from 1 not from 0
plt.ylabel("Model Score")
plt.xlabel("Number of features")
plt.title(f" Feature selection using RFECV \n Optimal number of features is {optimal_feature_count}(at score of{round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()

# Model Training
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Model Assessment
# predict on the test set
y_pred=regressor.predict(X_test)

#calculate r2 score
r_squared=r2_score(y_test, y_pred)

# Cross Validation
cv=KFold(n_splits = 4, shuffle= True, random_state=42)
cv_scores = cross_val_score(regressor, X_train, y_train,cv=cv, scoring='r2')  # default cv=5
cv_scores.mean()

# Calculate adjusted r2
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1-r_squared)*(num_data_points-1)/(num_data_points-num_input_vars-1)

# Extract model coefficient
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stat=pd.concat([input_variable_names, coefficients], axis=1)
summary_stat.columns = ["input_variable_names", "coefficients"]

# Extract model intercept
regressor.intercept_
print(regressor.intercept_)































