import pandas as pd
import numpy as np
from numpy import linalg as ln
import openpyxl

class LinerRegression:
    def __init__(self, X_variables : pd.DataFrame, Y_variable : pd.Series):        
        self.X = X_variables.to_numpy()
        self.Y = Y_variable.to_numpy()
        self.__estimate_coeff_()

    def __estimate_coeff_(self):
        self.X = np.insert(self.X, 0, 1.0, axis= 1)

        X_T = np.transpose(self.X)  # X^T       --> X transpose         
        XTX = X_T.dot(self.X)       # X^TX      --> X transpose dikali dengan matriks X
        X_invers = ln.inv(XTX)      # (X^TX)^-1 --> Invers dari matriks X^TX
        XTY = X_T.dot(self.Y)       # X^TY      --> X transpose dikali dengan matriks Y     # (X^TX)^-1 * X^TY    --> Inversi dari matriks X^TX dikali dengan matriks X^TY

    
        estimate_coef = X_invers.dot(XTY)           # bentuk nya -> [β0, β1, β2, ...., βx]
        self.intercept__ = estimate_coef[0]         #  β0                               -> intercept 
        self.coef__ = estimate_coef[1:].tolist()    # β1, β2, βx                        -> koefisien regresi 

    def predict(self, x : list[int | float]):
        if len(x) > len(self.coef__) or len(x) < len(self.coef__):
            raise ValueError("count of X predict should be same with count of X variables")
        
        predict_result = self.intercept__
        for i in range(len(x)):
            predict_result += (self.coef__[i] * x[i])
        return round(predict_result, 3)
    
    @staticmethod
    def accuracy(a: list[float], b: list[float]):
        dot_product = sum([a[i] * b[i] for i in range(len(a))])
        magnitude_a = np.sqrt(sum([x**2 for x in a]))
        magnitude_b = np.sqrt(sum([x**2 for x in b]))

        return (dot_product / (magnitude_a * magnitude_b)) * 100
    
    @staticmethod
    def R_squared(y_actual: list[float], y_predict: list[float]):
        avg = np.average(y_actual)

        SS_res = sum([(y_actual[i] - y_predict[i])**2 for i in range(len(y_actual))])
        SS_tot = sum([(y_actual[i] - avg)**2 for i in range(len(y_actual))])

        return (1 - (SS_res / SS_tot))
    


    



# data = {
#     'Luas_Tanah': [120, 150, 200, 250, 300],
#     'Jumlah_Kamar': [2, 3, 4, 4, 5],
#     'Harga_Rumah': [500, 600, 800, 1000, 1200]
# }

# data = pd.DataFrame(data)

# x = data[['Luas_Tanah', 'Jumlah_Kamar']]
# y = data['Harga_Rumah']

# model = LinerRegression(x, y)
# predict = model.predict(120, 2)
# print(LinerRegression.accuracy([predict], [500]))
# sales_data = pd.read_excel("UAS Project\dataset\Smartphone_sales_clean.xlsx")

# x_variables = sales_data[['Memory', 'Storage', 'Original Price', 'Discount']]
# y_variables = sales_data['Selling Price']



"""
    β = (X^T * X)^-1 * (X^T * Y) --> estimate coeff formula
    Y = β0 + β1X1 + β2X2 + ... + βnXn  -> model regresi


    Data Prediksi (A): [4.5, 3.0, 5.0, 2.0]
    Data Aktual (B): [4.0, 3.5, 4.5, 2.5]

    sum([a[i] * b[i] for i in range(len(a))])
    _____________________________________________________       --> Cosine Similarity (for calculating same accuration)
    sqrt(sum([x**2 for x in a])) * sqrt(sum([x**2 for x in b]))

"""

