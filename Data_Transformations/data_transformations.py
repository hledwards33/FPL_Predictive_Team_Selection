# The purpose of this code is to perform all necessary generic data pre-processing of the raw data.
# The output of this code should serve as the input to the predictive models.

# Import modules
import os
import pandas as pd


# declare path to saved data
data_path = os.getcwd() + "\Data\Raw_Data\cleaned_merged_seasons.csv"

# Read player data into pandas dataframe
df = pd.read_csv(data_path).iloc[:, 1:]

def assign_latest_team(dataframe, date_field, player_field, team_field):
    """
    This function assigns players a new team based on the latest team we have data for. i.e. we ignore transfers.
    :param dataframe: full player dataset
    :param date_field: name of the datetime column in dataframe
    :param player_field: name of the player name column in dataframe
    :param team_field: name of the player team field in the dataframe
    :return:
    """
    # Sort values by match date
    dataframe.sort_values(by=[date_field], ascending=False, inplace=True)

    # Create dataset of only name instance
    player_team = dataframe.drop_duplicates(subset=[player_field])[[player_field, team_field]]
    player_team.reset_index(inplace=True, drop=True)

    # Merge player_team dataset with main dataframe
    dataframe = pd.merge(dataframe, player_team, how='left', on=['name']).rename(columns={'team_x_y': 'team'})

    # Drop previous team_field
    dataframe.drop(team_field + "_x", axis=1, inplace=True)

    return dataframe

def data_transformations(dataframe):
    # Convert kickoff_time field to datetime object
    df['match_date'] = pd.to_datetime(
        df['kickoff_time'].apply(lambda x: x[:-10]) + " " + df['kickoff_time'].apply(lambda x: x[-9:-1]))
    df.drop('kickoff_time', inplace=True, axis=1)
