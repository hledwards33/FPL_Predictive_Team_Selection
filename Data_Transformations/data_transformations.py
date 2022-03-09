# The purpose of this code is to perform all necessary generic data pre-processing of the raw data.
# The output of this code should serve as the input to the predictive models.

# Import modules
import pandas as pd
import os


def dataset_creation(input_data_path, input_data, team_rating_data, output_data_path, date_field, player_field,
                     team_field):
    """
    This function creates separate datasets for each position based
    :param input_data_path: file path to the input datasets
    :param input_data: file name of full player dataset
    :param team_rating_data: file name of the past premier league table data
    :param output_data_path: file path to save final datasets to
    :param date_field: date field name within the full player dataset
    :param player_field: player name field within the full player dataset
    :param team_field: team name field within the full player dataset
    :return: there is nothing returned datasets are saved to the output_data_path
    """
    # Declare path to saved data and outputted data

    # Read player data into pandas dataframe
    dataframe = pd.read_csv(os.path.join(input_data_path, input_data), low_memory=False).iloc[:, 1:]

    # Create cleaned and transformed dataset for for use in model
    dataframe = data_transformations(dataframe, date_field, player_field, team_field, input_data_path, team_rating_data)

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

    # Save datasets that have been created
    goalkeepers.to_csv(os.path.join(output_data_path, 'goalkeepers.csv'), index=False)
    defenders.to_csv(os.path.join(output_data_path, 'defenders.csv'), index=False)
    attackers.to_csv(os.path.join(output_data_path, 'attackers.csv'), index=False)
    goalkeepers_correlated.to_csv(os.path.join(output_data_path, 'goalkeepers_correlated.csv'), index=False)
    defenders_correlated.to_csv(os.path.join(output_data_path, 'defenders_correlated.csv'), index=False)
    attackers_correlated.to_csv(os.path.join(output_data_path, 'attackers_correlated.csv'), index=False)


def data_transformations(dataframe, date_field, player_field, team_field, input_data_path, team_rating_data):
    """
    This function takes care of the data transformations to make the data usable in the predictive model
    :param dataframe: full player dataset
    :param date_field: date field name within the full player dataset
    :param player_field: player name field within the full player dataset
    :param team_field: team name field within the full player dataset
    :param input_data_path: file path to the input datasets
    :param team_rating_data: file name of the past premier league table data
    :return: dataframe with with transformed data fields ready to be inputted into model for training
    """
    # Convert kickoff_time field to datetime object
    dataframe['match_date'] = pd.to_datetime(
        dataframe['kickoff_time'].apply(lambda x: x[:-10]) + " " + dataframe['kickoff_time'].apply(lambda x: x[-9:-1]))
    #  Drop the previous match_date field
    dataframe.drop('kickoff_time', inplace=True, axis=1)

    #  Assign players their latest team using the assign_latest_team function
    dataframe = assign_latest_team(dataframe, date_field, player_field, team_field)

    # Rename a select few variables
    dataframe.rename(columns={'season_x': 'Season', 'team_x': 'team'}, inplace=True)

    # Dealing with missing home and away scores - drop rows that contain null data
    dataframe.dropna(axis=0, inplace=True)

    # Remove player data points where the player has not played any minutes
    dataframe = dataframe[dataframe['minutes'] != 0]

    # Apply team rating system to the dataset
    dataframe = team_rating_system(dataframe, input_data_path, team_rating_data, 'opponent')
    dataframe = team_rating_system(dataframe, input_data_path, team_rating_data, 'home')

    # Some home teams do not have team ranking due to us using most recent team in dataset
    dataframe = fill_null_team_rating(dataframe, input_data_path, team_rating_data)

    # Rewrite True/False was_home field to 1/0 field
    dataframe['was_home'] = dataframe['was_home'].apply(lambda x: 1 if x==True else 0)

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


def team_rating_system(dataframe, input_data_path, team_rating_data, home_or_opp_team: str):
    """
    This function applies the team quartile rating system to the input datasets
    :param dataframe: full player dataset
    :param input_data_path: file path to the input datasets
    :param team_rating_data: file name of the past premier league table data
    :param home_or_opp_team: home/opponent based on which teams are being scored
    :return: dataframe containing quartile scores for opponent/home team
    """
    # Read in the team ratings data
    team_ratings = pd.read_csv(os.path.join(input_data_path, team_rating_data))

    # Define the team name column to apply mapping on and output ranking column names
    if home_or_opp_team == 'home':
        team_col_name = 'team'
        ranking_col_name = 'team_rating'
    else:
        team_col_name = 'opp_team_name'
        ranking_col_name = 'opp_team_rating'

    # Apply quartile scoring to the team's rankings
    team_ratings[ranking_col_name] = team_ratings['Position'].apply(lambda x: quartile_scoring(x))

    # Drop Position column from team_ratings
    team_ratings.drop('Position', axis=1, inplace=True)

    # Rename column name for mapping/merge
    team_ratings.rename(columns={'Team': team_col_name}, inplace=True)

    # Merge the input dataframe with team_ratings on Season and team_col_name
    dataframe = pd.merge(dataframe, team_ratings, how='left', on=['Season', team_col_name])

    if home_or_opp_team != 'home':
        dataframe.drop([team_col_name, 'opponent_team'], axis=1, inplace=True)

    # Return dataset with team ratings
    return dataframe


def quartile_scoring(rank):
    """
    This function gives a score between 1 - 4 based on a teams ranks between 1 - 20.
    Scoring logic is score = ceiling(rank/5)
    :param rank: team's rank in a given season
    :return: score between 1 - 4 based on the team's rank
    """
    # Create scoring system based on rank
    if rank <= 5:
        return 1
    elif rank <= 10:
        return 2
    elif rank <= 15:
        return 3
    else:
        return 4


def fill_null_team_rating(dataframe, input_data_path, team_rating_data):
    """
    This function fills in the null home team ranking data points using the team's most recent ranking
    :param dataframe: full player dataset
    :param input_data_path: file path to the input datasets
    :param team_rating_data: file name of the past premier league table data
    :return: dataframe containing complete team_rating field (no null values)
    """
    # Read in the ratings data
    team_ratings = pd.read_csv(os.path.join(input_data_path, team_rating_data))

    # Sort team ratings by Season, latest to oldest
    team_ratings.sort_values(by=['Season', 'Position'], ascending=[False, True], inplace=True)

    # Keep only the latest team position available
    team_latest_ratings = team_ratings.drop_duplicates(subset=['Team'])[['Team', 'Position']]

    # Apply quartile scoring to team_latest_ratings
    team_latest_ratings['team_latest_rating'] = team_latest_ratings['Position'].apply(lambda x: quartile_scoring(x))

    # Drop Position column from team_latest_rating
    team_latest_ratings.drop('Position', axis=1, inplace=True)

    # Rename column name for mapping/merge
    team_latest_ratings.rename(columns={'Team': 'team'}, inplace=True)

    # Merge team_latest_ratings column to input dataframe
    dataframe = pd.merge(dataframe, team_latest_ratings, how='left', on=['team'])

    # Merge columns team_latest_ratings and team_ratings where team_ratings is nan
    dataframe['merged_col'] = dataframe['team_rating'].where(dataframe['team_rating'].notnull(),
                                                             dataframe['team_latest_rating'])

    # Drop non-merged (old) team ratings columns
    dataframe.drop(['team_rating', 'team_latest_rating'], axis=1, inplace=True)

    # Rename merge team rating columns
    dataframe.rename(columns={'merged_col': 'team_rating'}, inplace=True)

    return dataframe
