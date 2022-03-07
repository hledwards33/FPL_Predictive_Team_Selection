# Import modules
import pandas as pd
import os
# Import methods from within project
from Data_Transformations.data_transformations import data_transformations


def main():
    # declare path to saved data
    data_path = os.path.join(os.getcwd(), "Data", "Raw_Data", "cleaned_merged_seasons.csv")

    # Read player data into pandas dataframe
    dataframe = pd.read_csv(data_path, low_memory=False).iloc[:, 1:]
    date_field = 'match_date'
    player_field = 'name'
    team_field = 'team_x'

    df = data_transformations(dataframe, date_field, player_field, team_field)

    return df


if __name__ == "__main__":
    main()
