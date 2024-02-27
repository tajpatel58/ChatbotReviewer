import pandas as pd
from pathlib import Path
from pyairtable import Api


def create_airtable_api(airtable_token_path: Path) -> Api:
    with open(airtable_token_path, "r") as f:
        token = f.read()
    airtable_api = Api(token)
    return airtable_api


def airtable_to_dataframe(
    airtable_api: Api, table_id: str, base_id: str
) -> pd.DataFrame:
    records = airtable_api.table(base_id, table_id).all()
    data_df = pd.DataFrame.from_records([r["fields"] for r in records])
    return data_df


base_id = "appt3ILyWkGcmpceN"
reviews_table_id = "tblNnw8GEfd5mcdY3"
api = create_airtable_api("./token.txt")
reviews_df = airtable_to_dataframe(api, reviews_table_id, base_id)
print(reviews_df.head())
