import pandas as pd
from config.config import settings, engine
from db.db_model import RentApartments
from sqlalchemy import select


def load_data_from_db():
    query = select(RentApartments)
    return pd.read_sql(query, engine)