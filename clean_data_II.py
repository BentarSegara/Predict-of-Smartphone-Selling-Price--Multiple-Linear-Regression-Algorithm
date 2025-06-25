import pandas as pd
import numpy as np

def printDF(data_frame : pd.DataFrame, row=0) -> pd.DataFrame:
    if (row > 0): 
        data_frame = data_frame[:row]

    print(data_frame.to_string(index=False))

def get_memory(memory_in_string: str):
    unit = memory_in_string[len(memory_in_string) - 2:]
    memory = ''.join(x for x in memory_in_string if x.isdigit())
    memory = float(memory)    

    return memory*1000 if (unit != 'MB') else memory


Brands_details = pd.read_excel("UAS Project\dataset\Smartphone_sales.xlsx")
Brands = pd.DataFrame(Brands_details)

Brands['Memory'] = Brands['Memory'].apply(lambda memory_string : get_memory(memory_string))
Brands['Storage'] = Brands['Storage'].apply(lambda storage_string : get_memory(storage_string))


printDF(Brands, 10)
Brands.to_excel("UAS Project\dataset\Smartphone_sales_clean.xlsx", sheet_name='Smartphone_sales_clean', index=False, engine='openpyxl')