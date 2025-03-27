# import required package

import joblib
import pandas as pd
import numpy as np

# import pipeline
regressor=joblib.load("house_price_model.joblib")

# create new data
new_data=pd.DataFrame({"Address":["Dindoli", "Palanpur", "Jahangirabad", "Mota Varachha"],
                      "areaWithType" : ["Carpet Area", "Carpet Area", "Carpet Area", "Carpet Area"],
                      "square_feet":[644, 720, 748, 1000],
                      "transaction":["New Property", "New Property", "New Property", "New Property"],
                      "status":["still in construction", "still in construction", "still in construction", "Ready to Move"],
                      "furnishing":["Unfurnished", "Unfurnished", "Unfurnished", "Furnished"],
                      "price_per_sqft":[2891, 3200, 3235, 3185],
                      "BHK":[2, 2, 2, 3]})

# pass the new_data in and receive the prediction
regressor.predict(new_data)
