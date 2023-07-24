import pandas as pd
import plotly.graph_objects as go
import pytest

from visualization.visualization import TestDataVisualization


def test_test_data_visualization():
    """
    Test the TestDataVisualization class.

    This test case ensures that the TestDataVisualization class correctly generates a Plotly figure for visualizing the
    test data, mapped ideal functions, and deviations. It creates sample test data, empty mapped dataframes, and an
    empty test mappings dataframe to create the TestDataVisualization object. The test then calls the create_figure
    method to generate the figure and verifies the content and type of the figure.
    """
    # Create sample test data
    test_data = pd.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [0.5, 0.7, 0.9, 1.1]
    })

    # Create empty mapped dataframes and test mappings
    mapped_dataframes = {}
    test_mappings = pd.DataFrame()

    # Create the visualization object
    visualization = TestDataVisualization(test_data, mapped_dataframes, test_mappings)

    # Call the create_figure method
    fig = visualization.create_figure()

    # Verify the type of the figure
    assert isinstance(fig, go.Figure)

    # Verify the traces
    assert len(fig.data) == 1  # Only the test data scatter plot
    assert fig.data[0].mode == 'markers'
    assert fig.data[0].name == 'Test Data'
    assert fig.data[0].x.tolist() == [1, 2, 3, 4]
    assert fig.data[0].y.tolist() == [0.5, 0.7, 0.9, 1.1]


# Run the test
pytest.main(['-qq'])
