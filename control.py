import pandas as pd
import numpy as np

from model import LinerRegression

sales_data = pd.read_excel("UAS Project\dataset\Smartphone_sales_clean.xlsx")

x_training = sales_data[['Memory', 'Storage', 'Original Price', 'Discount']]
y_training = sales_data['Selling Price']                                    

model = LinerRegression(x_training, y_training)                             

# training #

# x_training = sales_data[['Memory', 'Storage', 'Original Price', 'Discount']].iloc[:2425]
# y_training = sales_data['Selling Price'].iloc[:2425]

# x_testing = sales_data[['Memory', 'Storage', 'Original Price', 'Discount']]
# y_testing = sales_data['Selling Price'].iloc[2425:]



# model = LinerRegression(x_training, y_training)
# predicts = []


# for i in range(2425, len(sales_data)):
#     predict = model.predict(x_testing.iloc[i].tolist())
#     predicts.append(predict)

# y = y_testing.tolist()

# accuracy = LinerRegression.accuracy(predicts, y_testing.tolist())
# print(f"Akurasi : {accuracy} %")
# print(model.R_squared(y, predicts))
