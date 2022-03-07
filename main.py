# Import modules
import pandas as pd
import os
# Import methods from within project
from Data_Transformations.data_transformations import data_transformations, position_datasets


def main():
    # declare path to saved data
    data_path = os.path.join(os.getcwd(), "Data", "Raw_Data", "cleaned_merged_seasons.csv")

    # Read player data into pandas dataframe
    dataframe = pd.read_csv(os.path.join(data_path,"cleaned_merged_seasons.csv"), low_memory=False).iloc[:, 1:]
    date_field = 'match_date'
    player_field = 'name'
    team_field = 'team_x'

    # Create cleaned and transformed dataset for for use in model
    dataframe = data_transformations(dataframe, date_field, player_field, team_field)

    # Create datasets filtered by player's position
    goalkeepers = position_datasets(dataframe, 'GK')
    defenders = position_datasets(dataframe, 'DEF')
    attackers = position_datasets(dataframe, 'MID', 'FWD')

    # Define the highly correlated variables for each positional dataset
    goalkeepers_corr_vars = ['Season', 'name', 'position', 'team', 'total_points', 'bps', 'clean_sheets', 'minutes',
                             'bonus', 'influence', 'ict_index', 'saves', 'value', 'selected', 'penalties_saved',
                             'transfers_in']
    defenders_corr_vars = ['Season', 'name', 'position', 'team', 'total_points', 'bps', 'clean_sheets', 'bonus',
                           'influence', 'ict_index', 'minutes', 'goals_scored', 'threat', 'assists', 'creativity',
                           'value', 'selected']
    attackers_corr_vars = ['Season', 'name', 'position', 'team', 'total_points', 'bps', 'influence', 'goals_scored',
                           'ict_index', 'bonus', 'threat', 'minutes', 'creativity', 'assists', 'value', 'clean_sheets',
                           'selected']

    goalkeepers_correlated = goalkeepers[goalkeepers_corr_vars]
    defenders_correlated = defenders[defenders_corr_vars]
    attackers_correlated = attackers[attackers_corr_vars]

    # Save datasets that have been created



if __name__ == "__main__":
    main()
