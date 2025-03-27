import pandas as pd
import numpy as np
import pickle

data=pd.read_excel("surat.xlsx")

# Data Cleaning

cat_cols=data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col]=data[col].str.strip()
    
#filter only Apartment type properties

data_for_model=data[data["Type"]=="Apartment"]

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

#save our files
pickle.dump(data_for_model, open("data_for_model.p","wb"))

