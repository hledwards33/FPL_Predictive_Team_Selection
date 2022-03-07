# The purpose of this code is to perform all necessary generic data pre-processing of the raw data.
# The output of this code should serve as the input to the predictive models.

# Import modules
import pandas as pd
import os


def dataset_creation(input_data_path, input_data, output_data_path, date_field, player_field, team_field):
    # Declare path to saved data and outputted data

    # Read player data into pandas dataframe
    dataframe = pd.read_csv(os.path.join(input_data_path, input_data), low_memory=False).iloc[:, 1:]

    # Create cleaned and transformed dataset for for use in model
    dataframe = data_transformations(dataframe, date_field, player_field, team_field)

    # Rename a select few variables
    dataframe.rename(columns={'season_x': 'Season', 'team_x': 'team'}, inplace=True)

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

    # Create datasets only containing correlated values
    goalkeepers_correlated = goalkeepers.reindex(columns=goalkeepers_corr_vars)
    defenders_correlated = defenders.reindex(columns=defenders_corr_vars)
    attackers_correlated = attackers.reindex(columns=attackers_corr_vars)

    print('goalkeepers', len(goalkeepers))
    print('defenders', len(defenders))
    print('attackers', len(attackers))
    print('goalkeepers_correlated', len(goalkeepers_correlated))
    print('defenders_correlated', len(defenders_correlated))
    print('attackers_correlated', len(attackers_correlated))

    # Save datasets that have been created
    goalkeepers.to_csv(os.path.join(output_data_path, 'goalkeepers.csv'), index=False)
    defenders.to_csv(os.path.join(output_data_path, 'defenders.csv'), index=False)
    attackers.to_csv(os.path.join(output_data_path, 'attackers.csv'), index=False)
    goalkeepers_correlated.to_csv(os.path.join(output_data_path, 'goalkeepers_correlated.csv'), index=False)
    defenders_correlated.to_csv(os.path.join(output_data_path, 'defenders_correlated.csv'), index=False)
    attackers_correlated.to_csv(os.path.join(output_data_path, 'attackers_correlated.csv'), index=False)


def data_transformations(dataframe, date_field, player_field, team_field):
    """
    This function takes care of the data transformations to make the data usable in the predictive model
    :param dataframe: full player dataset
    :param date_field: date field name within the full player dataset
    :param player_field: player name field within the full player dataset
    :param team_field: team name field within the full player dataset
    :return: returns
    """
    # Convert kickoff_time field to datetime object
    dataframe['match_date'] = pd.to_datetime(
        dataframe['kickoff_time'].apply(lambda x: x[:-10]) + " " + dataframe['kickoff_time'].apply(lambda x: x[-9:-1]))
    #  Drop the previous match_date field
    dataframe.drop('kickoff_time', inplace=True, axis=1)

    #  Assign players their latest team using the assign_latest_team function
    dataframe = assign_latest_team(dataframe, date_field, player_field, team_field)

    # Dealing with missing home and away scores - drop rows that contain null data
    dataframe.dropna(axis=0, inplace=True)

    # Remove player data points where the player has not played any minutes
    dataframe = dataframe[dataframe['minutes'] != 0]

    # TODO: add in team rating system to the dataset

    # Return the transformed dataset
    return dataframe


def assign_latest_team(dataframe, date_field, player_field, team_field):
    """
    This function assigns players a new team based on the latest team for which we have data. i.e. we ignore transfers.
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


def position_datasets(dataframe, position1, position2=None):
    """
    This function filters the input dataframe to only include player of the chosen positions and outputs the resulting
    dataframe
    :param dataframe: full player dataset
    :param position1: first chosen position to be included in output dataframe
    :param position2: second chosen position to be included in output dataframe - optional argument
    :return: dataframe containing only rows of players in desired positions
    """
    # Create a dataset from the main based on desired positions
    if position2 is None:
        position_dataset = dataframe[dataframe['position'] == position1]
    else:
        position_dataset = dataframe[(dataframe['position'] == position1) | (dataframe['position'] == position2)]

    # Return the filtered datasets
    return position_dataset
