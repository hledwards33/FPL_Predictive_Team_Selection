# Import modules
import os
# Import methods from within project
from Data_Transformations.data_transformations import dataset_creation


def main():
    input_data_path = os.path.join(os.getcwd(), "Data", "Raw_Data")
    input_data = "cleaned_merged_seasons.csv"
    team_rating_data = "EPL_Tables.csv"
    output_data_path = os.path.join(os.getcwd(), "Data", "Transformed_Data")
    date_field = 'match_date'
    player_field = 'name'
    team_field = 'team_x'

    dataset_creation(input_data_path, input_data, team_rating_data, output_data_path, date_field, player_field,
                     team_field)


if __name__ == "__main__":
    main()
