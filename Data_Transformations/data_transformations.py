import os
import pandas as pd

data_path = os.path.dirname(os.getcwd()) + "\Data\Raw_Data\cleaned_merged_seasons.csv"

df = pd.read_csv(data_path).iloc[:, 1:]

print('End')

