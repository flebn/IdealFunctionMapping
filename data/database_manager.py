from sqlalchemy import create_engine, Column, Integer, Float, String, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class TrainingData(Base):
    """Class representing the 'training_data' table in the database.

    Attributes:
        x (float): The x value.
        y1-y4 (float): The y values for each training data point.
    """

    __tablename__ = 'training_data'

    x = Column(Float, primary_key=True)
    # Dynamically create the columns y1, y2, y3, y4
    for i in range(1, 5):
        locals()[f"y{i}"] = Column(Float)

    def __init__(self, **kwargs):
        super().__init__()  # Call the __init__ method of the superclass
        for i in range(1, 5):
            setattr(self, f'y{i}', kwargs.get(f'y{i}', None))

    __table_args__ = (
        PrimaryKeyConstraint('x', name='training_data_pk'),
    )


class IdealFunctions(Base):
    """Class representing the 'ideal_functions' table in the database.

    Attributes:
        x (float): The x value.
        y1-y50 (float): The y values for each ideal function.
    """

    __tablename__ = 'ideal_functions'

    x = Column(Float, primary_key=True)
    # Dynamically create the columns y1, y2, y3, ..., y50
    for i in range(1, 51):
        locals()[f"y{i}"] = Column(Float)

    def __init__(self, x, **kwargs):
        super().__init__()  # Call the __init__ method of the superclass
        self.x = x
        for i in range(1, 51):
            setattr(self, f'y{i}', kwargs.get(f'y{i}', None))

    __table_args__ = (
        PrimaryKeyConstraint('x', name='ideal_functions_pk'),
    )


class TestMapping(Base):
    """Class representing the 'test_mapping' table in the database.

    Attributes:
        id (int): The ID of the test mapping.
        x (float): The x value of the test data point.
        y (float): The y value of the test data point.
        Deviation (float): The deviation value of the test data point.
        IdealFunction (str): The name of the ideal function to which the test data point is mapped.
    """

    __tablename__ = 'test_mapping'

    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    Deviation = Column(Float)
    IdealFunction = Column(String)


class DatabaseManager:
    """Class for managing the SQLite database.

    Attributes:
        db_file (str): The path to the SQLite database file.
        engine (Engine): The SQLAlchemy engine object for database interaction.
        session (Session): The SQLAlchemy session object for querying the database.
    """

    def __init__(self, db_file):
        self.db_file = db_file
        self.engine = create_engine(f'sqlite:///{db_file}')
        Base.metadata.bind = self.engine
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(self.engine)

    def load_training_data(self, table_name, training_data):
        """Load training data into the specified table in the database.

        Args:
            table_name (str): The name of the table.
            training_data (DataFrame): Training data as a Pandas DataFrame.

        Returns:
            None
        """

        # Drop the table if it exists
        if self.engine.has_table(table_name):
            Base.metadata.tables[table_name].drop(self.engine)

        # Create the table
        Base.metadata.tables[table_name].create(self.engine)

        # Remove the index from the DataFrame
        training_data = training_data.reset_index(drop=True)

        # Insert the data into the table
        with self.engine.begin() as connection:
            for row in training_data.itertuples(index=False):
                x = row[0]  # First column contains x values
                kwargs = {f'y{i}': value for i, value in enumerate(row[1:], start=1)}
                training_instance = TrainingData(x=x, **kwargs)
                connection.execute(training_instance.__table__.insert().values(x=x, **kwargs))

        self.session.commit()

    def load_ideal_functions(self, data):
        """Load ideal functions data into the 'ideal_functions' table in the database.

        Args:
            data (list): List of rows, where each row contains the x value and y values of an ideal function.

        Returns:
            None
        """

        # Drop the table if it exists
        if self.engine.has_table(IdealFunctions.__tablename__):
            Base.metadata.tables[IdealFunctions.__tablename__].drop(self.engine)

        # Create the table
        Base.metadata.tables[IdealFunctions.__tablename__].create(self.engine)

        for row in data:
            if len(row) != 51:
                continue  # Skip rows that don't have the expected number of elements
            x = row[0]
            kwargs = {f'y{i}': row[i] for i in range(1, 51)}
            ideal_function = IdealFunctions(x=x, **kwargs)
            self.session.add(ideal_function)

        self.session.commit()

    def load_test_mapping(self, data):
        """Load test data mapping into the 'test_mapping' table in the database.

        Args:
            data (list): List of rows, where each row contains the x, y, deviation, and ideal function name.

        Returns:
            None
        """

        # Drop the table if it exists
        if self.engine.has_table(TestMapping.__tablename__):
            Base.metadata.tables[TestMapping.__tablename__].drop(self.engine)

        # Create the table
        Base.metadata.tables[TestMapping.__tablename__].create(self.engine)

        for row in data:
            x = row[0]
            y = row[1]
            deviation = row[2]
            ideal_function = row[3]
            test_mapping = TestMapping(x=x, y=y, Deviation=deviation, IdealFunction=ideal_function)
            self.session.add(test_mapping)

        self.session.commit()

    def fetch_data(self, table_name):
        """Fetch data from the specified table in the database.

        Args:
            table_name (class): The class representing the table.

        Returns:
            list: A list of objects representing the rows in the table.
        """
        return self.session.query(table_name).all()

    def close(self):
        """Close the database session."""
        self.session.close()
