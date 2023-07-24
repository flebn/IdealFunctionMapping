from typing import Dict

import numpy as np
import pandas as pd


class Functions:
    """Contains methods for calculating deviations and mapping test data."""

    @staticmethod
    def map_training_to_ideal_functions(training_data: pd.DataFrame, ideal_functions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the deviations between training data and ideal functions, and returns a DataFrame containing
        the mapping between training data columns and best fitting ideal function columns.

        Args:
            training_data (DataFrame): Training data as a Pandas DataFrame.
            ideal_functions (DataFrame): Ideal functions as a Pandas DataFrame.

        Returns:
            DataFrame: DataFrame containing the mapping between training data columns and best fitting ideal function columns.

        This method takes two dataframes, `training_data` and `ideal_functions`, and calculates the deviations between
        the two datasets for each column in the training data. For each training data column, the best fitting ideal
        function column is determined based on the least squared difference between the training data values and the
        ideal function values. The result is a dataframe containing the mapping between training data columns and their
        corresponding best fitting ideal function columns.
        """
        mappings = pd.DataFrame(columns=["TrainingData", "IdealFunction"])
        training_columns = training_data.columns[
                           1:]  # Exclude the 'x' column from training_data. Represents the input 'x' values for the functions

        try:
            # Align the data frames based on the 'x' column
            aligned_training_data = training_data.set_index('x')
            aligned_ideal_functions = ideal_functions.set_index('x')
        except KeyError as e:
            raise KeyError(f"Invalid column name: {e}") from err

        for train_column in training_columns:
            try:
                train_values = aligned_training_data[train_column].values
            except KeyError as e:
                raise KeyError(f"Invalid column name: {e}") from e

            best_fit_function = None
            best_fit_deviation = float("inf")

            for ideal_column in aligned_ideal_functions.columns:
                try:
                    ideal_values = aligned_ideal_functions[ideal_column].values
                except KeyError as e:
                    raise KeyError(f"Invalid column name: {e}") from err

                try:
                    squared_diff = ((train_values - ideal_values) ** 2).sum()
                except ValueError as e:
                    raise ValueError(
                        "Mismatch in array shapes. Ensure training_data and ideal_functions have the same length.") from e

                if squared_diff < best_fit_deviation:
                    best_fit_function = ideal_column
                    best_fit_deviation = squared_diff

            mappings = pd.concat(
                [
                    mappings,
                    pd.DataFrame({"TrainingData": [train_column], "IdealFunction": [best_fit_function]}),
                ],
                ignore_index=True,
            )

        return mappings

    @staticmethod
    def create_mapped_dataframes(mappings: pd.DataFrame, ideal_functions: pd.DataFrame, training_data: pd.DataFrame) -> \
    Dict[str, pd.DataFrame]:
        """
        Creates separate dataframes for each mapped ideal function by extracting relevant columns from the input
        dataframes to perform calculations on them.

        Args:
            mappings (DataFrame): Mapping between training data columns and best fitting ideal function columns.
            ideal_functions (DataFrame): Ideal functions as a Pandas DataFrame.
            training_data (DataFrame): Training data as a Pandas DataFrame.

        Returns:
            Dict[str, DataFrame]: Dictionary of dataframes containing the mapped ideal functions.

        This method takes the mappings dataframe, ideal functions dataframe, and training data dataframe as inputs. It creates
        separate dataframes for each mapped ideal function based on the mappings provided. For each mapped ideal function,
        the method extracts the relevant 'x', 'y_ideal', and 'y_train' columns from the input dataframes to perform further
        calculations. The 'x' column is extracted from the training data, the 'y_ideal' column is extracted from the ideal
        functions dataframe based on the mapped ideal function, and the 'y_train' column is extracted from the training data
        based on the corresponding training data column in the mappings. The method then calculates the 'y_diff' column by
        taking the difference between the 'y_ideal' and 'y_train' columns. The result is a dictionary of dataframes, where
        each dataframe corresponds to a mapped ideal function and contains the 'x', 'y_ideal', 'y_train', and 'y_diff'
        columns.
        """
        mapped_dataframes = {}

        for _, row in mappings.iterrows():
            training_data_column = row["TrainingData"]
            ideal_function_column = row["IdealFunction"]

            if ideal_function_column not in ideal_functions.columns:
                continue

            try:
                mapped_dataframe = pd.DataFrame()
                mapped_dataframe["x"] = training_data["x"]  # Use 'x' column from training_data instead
                mapped_dataframe["y_ideal"] = ideal_functions[ideal_function_column]
                mapped_dataframe["y_train"] = training_data[training_data_column]
                mapped_dataframe["y_diff"] = mapped_dataframe["y_ideal"] - mapped_dataframe["y_train"]
            except KeyError as e:
                raise KeyError(f"Invalid column name: {e}") from e

            mapped_dataframes[ideal_function_column] = mapped_dataframe

        return mapped_dataframes

    @staticmethod
    def map_test_data(test_data, mapped_dataframes):
        """
        Map test data to ideal functions and calculate deviations based on the revised deviation criterion.

        Args:
            test_data (pd.DataFrame): Test data as a Pandas DataFrame.
            mapped_dataframes (dict): Dictionary of dataframes containing the mapped ideal functions.

        Returns:
            pd.DataFrame: DataFrame containing the mappings and deviations for the test data.

        This method takes the test data as a Pandas DataFrame and a dictionary of dataframes `mapped_dataframes`, where each
        dataframe represents a mapped ideal function. The method maps the test data to the ideal functions and calculates
        deviations based on the revised deviation criterion. For each test data point, the method iterates through the
        mapped_dataframes to find the best fitting ideal function. The method calculates the deviation between the test data
        and the mapped ideal functions. The revised deviation criterion is applied, where the deviation between the training
        data and the mapped ideal function is compared to the deviation between the test data and the same mapped ideal
        function. If the deviation between the training data is greater than or equal to the deviation between the test data
        multiplied by the square root of 2, the test data is mapped to that ideal function. The mappings and deviations are
        stored in a new DataFrame `mapped_test_data`, which is returned as the output of the method.
        """
        mapped_test_data = pd.DataFrame(columns=["x", "y", "Deviation", "IdealFunction"])

        for _, test_case in test_data.iterrows():
            x_value = test_case["x"]
            y_value = test_case["y"]

            existing_max_deviation = np.nan
            best_fit_function = None
            best_fit_deviation = np.nan

            for function_name, dataframe in mapped_dataframes.items():
                deviation_train = dataframe["y_diff"].max()
                deviation_test = np.abs(y_value - dataframe.loc[dataframe["x"] == x_value, "y_ideal"].values[0])

                if deviation_train >= deviation_test * np.sqrt(2):
                    if np.isnan(existing_max_deviation) or deviation_test > existing_max_deviation:
                        existing_max_deviation = deviation_test
                        best_fit_function = function_name
                        best_fit_deviation = deviation_train

            mapped_test_data = pd.concat(
                [
                    mapped_test_data,
                    pd.DataFrame({"x": [x_value], "y": [y_value], "Deviation": [best_fit_deviation],
                                  "IdealFunction": [best_fit_function]}),
                ],
                ignore_index=True,
            )

        return mapped_test_data



