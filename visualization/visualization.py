import plotly.graph_objects as go
from plotly.subplots import make_subplots
from util.error_handling import VisualizationError


class BaseVisualization:
    """Base class for visualizations.

    Attributes:
        data (DataFrame): The data to be visualized.
    """

    def __init__(self, data):
        self.data = data

    def show(self):
        """Create and show the visualization figure."""
        fig = self.create_figure()
        fig.show()

    def create_figure(self):
        """Create the visualization figure.

        Raises:
            VisualizationError: This method should be overridden in the derived classes.
        """
        raise VisualizationError()


class TrainingDataVisualization(BaseVisualization):
    """Class for visualizing training data.

    This class creates a scatter plot of the training data.

    Attributes:
        data (DataFrame): The training data to be visualized.
    """

    def create_figure(self):
        """Create the scatter plot of the training data.

        Returns:
            go.Figure: The scatter plot figure.
        """
        fig = go.Figure()

        for column in self.data.columns[1:]:
            fig.add_trace(go.Scatter(x=self.data['x'], y=self.data[column], mode='markers', name=column))

        fig.update_layout(
            title='Training Data',
            xaxis_title='x',
            yaxis_title='y',
            showlegend=True
        )

        return fig


class IdealFunctionsVisualization(BaseVisualization):
    """Class for visualizing ideal functions.

    This class creates a scatter plot of the ideal functions.

    Attributes:
        data (DataFrame): The ideal functions to be visualized.
    """

    def create_figure(self):
        """Create the scatter plot of the ideal functions.

        Returns:
            go.Figure: The scatter plot figure.
        """
        fig = go.Figure()

        for column in self.data.columns:
            fig.add_trace(go.Scatter(x=self.data['x'], y=self.data[column], mode='markers', name=column))

        fig.update_layout(title='Ideal Functions', xaxis_title='x', yaxis_title='y')
        fig.update_yaxes(tickformat='.4f')

        return fig


class MappedDataframesVisualization(BaseVisualization):
    """Class for visualizing mapped training data to ideal functions.

    This class creates a subplots figure with multiple scatter and line plots.

    Attributes:
        data (dict): A dictionary of DataFrames containing mapped data for each ideal function.
    """

    def create_figure(self):
        """Create the subplots figure with multiple scatter and line plots.

        Returns:
            go.Figure: The subplots figure.
        """
        num_plots = len(self.data)
        subplot_titles = [f"Training Data: y{i + 1}, Ideal Function: {name}" for i, name in enumerate(self.data.keys())]
        fig = make_subplots(rows=num_plots, cols=1, subplot_titles=subplot_titles)

        for i, (function_name, dataframe) in enumerate(self.data.items()):
            fig.add_trace(go.Scatter(x=dataframe['x'], y=dataframe['y_ideal'], mode='lines',
                                     name=f'Ideal Function: {function_name}'), row=i + 1, col=1)
            fig.add_trace(
                go.Scatter(x=dataframe['x'], y=dataframe['y_train'], mode='markers', name=f'Training Data: y{i + 1}',
                           marker=dict(size=4)), row=i + 1, col=1)

        fig.update_layout(height=1500, width=1500, title_text="Mapped Training Data to Ideal Functions")

        return fig


class TestDataVisualization(BaseVisualization):
    """Class for visualizing test data and mapped ideal functions.

    This class creates a subplots figure with a scatter plot of test data and line plots for mapped ideal functions.

    Attributes:
        test_data (DataFrame): The test data to be visualized.
        mapped_dataframes (dict): A dictionary of DataFrames containing mapped data for each ideal function.
        test_mappings (DataFrame): The mappings of test data to ideal functions.
    """

    def __init__(self, test_data, mapped_dataframes, test_mappings):
        super().__init__(test_data)
        self.mapped_dataframes = mapped_dataframes
        self.test_mappings = test_mappings

    def create_figure(self):
        """Create the subplots figure with a scatter plot of test data and line plots for mapped ideal functions.

        Returns:
            go.Figure: The subplots figure.
        """
        num_plots = 1
        subplot_titles = ["Test Data"]
        fig = make_subplots(rows=num_plots, cols=1, subplot_titles=subplot_titles)

        # Add scatter plot for test data
        fig.add_trace(
            go.Scatter(x=self.data['x'], y=self.data['y'], mode='markers', name='Test Data', marker=dict(color='blue')),
            row=1, col=1)

        # Map ideal function names to colors
        color_mapping = {function_name: color for function_name, color in
                         zip(self.mapped_dataframes.keys(), ['red', 'DarkCyan', 'MediumPurple', 'Orange'])}

        # Add scatter plots for mapped ideal functions
        for function_name, dataframe in self.mapped_dataframes.items():
            color = color_mapping.get(function_name, 'lightgray')
            fig.add_trace(go.Scatter(x=dataframe['x'], y=dataframe['y_ideal'], mode='lines',
                                     name=f'Ideal Function: {function_name}', line=dict(color=color)), row=1, col=1)

        # Color the test data points based on the mapped ideal function
        for _, row in self.test_mappings.iterrows():
            ideal_function = row['IdealFunction']
            color = color_mapping.get(ideal_function, 'lightgray')
            fig.add_trace(go.Scatter(x=[row['x']], y=[row['y']], mode='markers', name=f'Test Data ({ideal_function})',
                                     marker=dict(color=color, symbol='circle', size=8,
                                                 line=dict(color='white', width=1))), row=1, col=1)

        return fig
