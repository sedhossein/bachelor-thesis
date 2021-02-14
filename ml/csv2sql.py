# Imports
import pandas as pd
from sqlalchemy import create_engine

# This CSV doesn't have a header so pass
# column names as an argument
columns = [
    "id1",
    "id2",
    "username",
    "short_description",
    "predict"
]

# Instantiate sqlachemy.create_engine object
engine = create_engine('postgresql://dashboard:dashboard@localhost:5432/dashboard')

# Create an iterable that will read "chunksize=1000" rows
# at a time from the CSV file
for df in pd.read_csv("data/predicted-tweets.csv", names=columns, chunksize=1000):
    df.drop(columns=['id1', 'id2'])
    df.to_sql(
        'tweets', #table name
        engine,
        index=True,
        if_exists='append'  # if the table already exists, append this data
    )
