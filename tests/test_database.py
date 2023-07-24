import pandas as pd
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.database_manager import DatabaseManager, TrainingData


def test_load_training_data():
    """
    Test the loading of training data into the database.

    This test creates an in-memory SQLite database for testing purposes. It loads test data into the database using the
    load_training_data method of the DatabaseManager class. Then, it fetches the data from the database and compares it
    with the original input data to ensure that the loading was successful.
    """
    # Create an in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)

    # Create a test instance of the DatabaseManager
    db_manager = DatabaseManager(db_file=":memory:")
    db_manager.engine = engine
    db_manager.session = Session()

    # Define the input data for the TrainingData table
    input_data = pd.DataFrame({'x': [1, 2, 3, 4, 5],
                               'y1': [0.1, 0.2, 0.3, 0.4, 0.5],
                               'y2': [1.1, 1.2, 1.3, 1.4, 1.5]})

    # Call the load_training_data method
    db_manager.load_training_data(table_name="training_data", training_data=input_data)

    # Fetch the data from the database
    with Session() as session:
        fetched_data = session.query(TrainingData).all()

    # Verify that the fetched data matches the original input data
    assert len(fetched_data) == len(input_data)

    for i, row in enumerate(fetched_data):
        assert row.x == input_data.loc[i, "x"]
        for column in input_data.columns[1:]:
            assert getattr(row, column) == input_data.loc[i, column]

    # Drop the TrainingData table after testing
    TrainingData.__table__.drop(bind=engine)

# Run the test
pytest.main(['-qq'])
