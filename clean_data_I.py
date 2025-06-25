import pandas as pd

Brands_details = pd.read_excel("UAS Project\dataset\Sales.xlsx", sheet_name='needed')
Brands = pd.DataFrame(Brands_details)
Brands = Brands.dropna().reset_index(drop=True)

print(Brands.head())

Brands.to_excel("UAS Project\dataset\Smartphone_sales.xlsx", sheet_name="Smartphone_sales", index=False, engine="openpyxl")
