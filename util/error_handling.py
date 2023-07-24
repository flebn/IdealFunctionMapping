class BaseErrorHandler(Exception):
    """
    Base class for custom error handlers.

    This class serves as the base exception class for custom error handlers. It can be inherited to create specific
    error classes for different parts of the codebase.
    """

    def __init__(self, message="An error occurred."):
        """
        Initialize the BaseErrorHandler instance.

        Args:
            message (str, optional): The error message. Defaults to "An error occurred."
        """
        self.message = message
        super().__init__(self.message)


class VisualizationError(BaseErrorHandler):
    """
    Exception raised for visualization-related errors.

    This class represents an exception that can be raised for errors related to visualizations. It inherits from the
    BaseErrorHandler class, allowing customization of error messages for visualization issues.
    """

    def __init__(self, message="An error occurred during visualization."):
        """
        Initialize the VisualizationError instance.

        Args:
            message (str, optional): The error message. Defaults to "An error occurred during visualization."
        """
        super().__init__(message)
