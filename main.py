import pandas as pd
from tabulate import tabulate

from data.database_manager import DatabaseManager, TrainingData, IdealFunctions, TestMapping
from functions.functions import Functions
from visualization.visualization import TrainingDataVisualization, IdealFunctionsVisualization, \
    MappedDataframesVisualization, TestDataVisualization
from util.error_handling import VisualizationError


def run():
    # Create database manager
    db_manager = DatabaseManager("data/database.db")
    db_manager.create_tables()

    # Load training data into the database
    training_data = pd.read_csv("data/raw_data/train.csv")
    db_manager.load_training_data("training_data", training_data)

    training_data_from_db = db_manager.fetch_data(TrainingData)
    training_data_list = [[getattr(row, column) for column in TrainingData.__table__.columns.keys()] for row in
                          training_data_from_db]
    print("Training Data has been loaded into database:")
    print(tabulate(training_data_list, headers=TrainingData.__table__.columns.keys(), tablefmt="pretty"))

    # Load ideal functions into the database
    ideal_functions = pd.read_csv("data/raw_data/ideal.csv")
    db_manager.load_ideal_functions(ideal_functions.values.tolist())

    ideal_functions_from_db = db_manager.fetch_data(IdealFunctions)
    ideal_functions_list = [[getattr(row, column) for column in IdealFunctions.__table__.columns.keys()] for row in
                            ideal_functions_from_db]
    print("Ideal Functions has been loaded into database:")
    print(tabulate(ideal_functions_list, headers=IdealFunctions.__table__.columns.keys(), tablefmt="pretty"))

    # Calculate deviations and map the training data to the ideal functions
    mapping = Functions.map_training_to_ideal_functions(training_data, ideal_functions)
    print("Ideal functions mapped to training data:")
    print(mapping)
    mapped_dataframes = Functions.create_mapped_dataframes(mapping, ideal_functions, training_data)

    # Load test data
    test_data = pd.read_csv("data/raw_data/test.csv")

    # Map test data
    mapped_test_data = Functions.map_test_data(test_data, mapped_dataframes)

    # Load test mappings into the database
    db_manager.load_test_mapping(mapped_test_data.values.tolist())
    test_mapping_from_db = db_manager.fetch_data(TestMapping)
    test_mapping_list = [[getattr(row, column) for column in TestMapping.__table__.columns.keys()] for row in
                         test_mapping_from_db]
    print("Test Mapping has been loaded into database:")
    print(tabulate(test_mapping_list, headers=TestMapping.__table__.columns.keys(), tablefmt="pretty"))

    # Visualize the data
    try:
        training_data_vis = TrainingDataVisualization(training_data)
        training_data_vis.show()

        ideal_functions_vis = IdealFunctionsVisualization(ideal_functions)
        ideal_functions_vis.show()

        mapped_dataframes_vis = MappedDataframesVisualization(mapped_dataframes)
        mapped_dataframes_vis.show()

        test_data_vis = TestDataVisualization(test_data, mapped_dataframes, mapped_test_data)
        test_data_vis.show()

    except VisualizationError as e:
        print(e.message)

    # Close the database connection
    db_manager.close()


if __name__ == "__main__":
    run()
