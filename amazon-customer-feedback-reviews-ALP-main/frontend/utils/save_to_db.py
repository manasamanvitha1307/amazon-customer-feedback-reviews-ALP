import pandas as pd
from db.database import engine

def save_results_to_db(df: pd.DataFrame, table_name="review_analysis"):
    df.to_sql(table_name, con=engine, if_exists="append", index=False)