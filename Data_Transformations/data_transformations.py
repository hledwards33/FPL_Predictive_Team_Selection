# The purpose of this code is to perform all necessary generic data pre-processing of the raw data.
# The output of this code should serve as the input to the predictive models.

# Import modules
import pandas as pd


def assign_latest_team(dataframe, date_field, player_field, team_field):
    """
    This function assigns players a new team based on the latest team we have data for. i.e. we ignore transfers.
    :param dataframe: full player dataset
    :param date_field: name of the datetime column in dataframe
    :param player_field: name of the player name column in dataframe
    :param team_field: name of the player team field in the dataframe
    :return: dataframe with new team field and prior team field dropped
    """
    # Sort values by match date
    dataframe.sort_values(by=[date_field], ascending=False, inplace=True)

    # Create dataset containing players names and their mose recent teams contained in the dataframe
    player_team = dataframe.drop_duplicates(subset=[player_field])[[player_field, team_field]]
    # Reset the index for the player_team dataset
    player_team.reset_index(inplace=True, drop=True)

    # Drop previous team_field from the main dataframe
    dataframe.drop(team_field, axis=1, inplace=True)

    # Merge player_team dataset with main dataframe
    dataframe = pd.merge(dataframe, player_team, how='left', on=['name'])

    # Return the modified dataframe
    return dataframe


def data_transformations(dataframe, date_field, player_field, team_field):
    # Convert kickoff_time field to datetime object
    dataframe['match_date'] = pd.to_datetime(
        dataframe['kickoff_time'].apply(lambda x: x[:-10]) + " " + dataframe['kickoff_time'].apply(lambda x: x[-9:-1]))
    #  Drop the previous match_date field
    dataframe.drop('kickoff_time', inplace=True, axis=1)

    #  Assign players their latest team using the assign_latest_team function
    dataframe = assign_latest_team(dataframe, date_field, player_field, team_field)

    # Dealing with missing home and away scores - drop rows that contain null data
    dataframe.dropna(axis=0, inplace=True)

    # Return the transformed dataset
    return dataframe

