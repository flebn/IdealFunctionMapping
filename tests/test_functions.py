import pandas as pd
import pytest

from functions.functions import Functions


def test_map_training_to_ideal_functions():
    """
    Test the mapping of training data to ideal functions.

    This test case ensures that the mapping between training data and ideal functions is calculated correctly. It uses
    sample training data and ideal functions dataframes to check if the expected mappings match the calculated mappings
    from the `map_training_to_ideal_functions` function.
    """
    # Create sample training data and ideal functions dataframes
    training_data = pd.DataFrame({'x': [1, 2, 3, 4, 5],
                                  'y1': [0.1, 0.2, 0.3, 0.4, 0.5],
                                  'y2': [1.1, 1.2, 1.3, 1.4, 1.5],
                                  'y3': [2.1, 2.2, 2.3, 2.4, 2.5]})

    ideal_functions = pd.DataFrame({'x': [1, 2, 3, 4, 5],
                                    'y1': [0.05, 0.15, 0.25, 0.35, 0.45],
                                    'y2': [1.05, 1.15, 1.25, 1.35, 1.45],
                                    'y3': [2.05, 2.15, 2.25, 2.35, 2.45],
                                    'y4': [3.05, 3.15, 3.25, 3.35, 3.45]})

    # Define the expected mappings
    expected_mappings = pd.DataFrame({'TrainingData': ['y1', 'y2', 'y3'],
                                      'IdealFunction': ['y1', 'y2', 'y3']})

    # Call the function and compare the returned mappings with the expected mappings
    mappings = Functions.map_training_to_ideal_functions(training_data, ideal_functions)
    pd.testing.assert_frame_equal(mappings, expected_mappings)


def test_create_mapped_dataframes():
    """
    Test the creation of mapped dataframes.

    This test case verifies the behavior of the `create_mapped_dataframes` function. The function should create separate
    dataframes for each mapped ideal function and ensure the columns are correctly extracted and calculated.
    """
    # Create a sample mappings dataframe
    mappings = pd.DataFrame({
        "TrainingData": ["y1", "y2"],
        "IdealFunction": ["ideal1", "ideal2"]
    })

    # Create sample ideal functions dataframe
    ideal_functions = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "ideal1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "ideal2": [1.1, 1.2, 1.3, 1.4, 1.5]
    })

    # Create sample training data dataframe
    training_data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "y2": [1.1, 1.2, 1.3, 1.4, 1.5]
    })

    # Call the function to create the mapped dataframes
    mapped_dataframes = Functions.create_mapped_dataframes(mappings, ideal_functions, training_data)

    # Verify the created mapped dataframes
    assert isinstance(mapped_dataframes, dict)
    assert len(mapped_dataframes) == 2

    mapped_dataframe1 = mapped_dataframes["ideal1"]
    assert isinstance(mapped_dataframe1, pd.DataFrame)
    assert set(mapped_dataframe1.columns) == {"x", "y_ideal", "y_train", "y_diff"}
    assert (mapped_dataframe1["x"] == [1, 2, 3, 4, 5]).all()
    assert (mapped_dataframe1["y_ideal"] == [0.1, 0.2, 0.3, 0.4, 0.5]).all()
    assert (mapped_dataframe1["y_train"] == [0.1, 0.2, 0.3, 0.4, 0.5]).all()
    assert (mapped_dataframe1["y_diff"] == [0.0, 0.0, 0.0, 0.0, 0.0]).all()

    mapped_dataframe2 = mapped_dataframes["ideal2"]
    assert isinstance(mapped_dataframe2, pd.DataFrame)
    assert set(mapped_dataframe2.columns) == {"x", "y_ideal", "y_train", "y_diff"}
    assert (mapped_dataframe2["x"] == [1, 2, 3, 4, 5]).all()
    assert (mapped_dataframe2["y_ideal"] == [1.1, 1.2, 1.3, 1.4, 1.5]).all()
    assert (mapped_dataframe2["y_train"] == [1.1, 1.2, 1.3, 1.4, 1.5]).all()
    assert (mapped_dataframe2["y_diff"] == [0.0, 0.0, 0.0, 0.0, 0.0]).all()


# Run the test
pytest.main(['-qq'])
