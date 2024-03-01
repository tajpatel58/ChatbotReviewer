import json
import pandas as pd
from pathlib import Path
from pyairtable import Api
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.airtable.api_schemas import Review

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_airtable_api(airtable_token_path: Path) -> Api:
    with open(airtable_token_path, "r") as f:
        token = f.read()
    airtable_api = Api(token)
    return airtable_api


def load_airtable_mapping_dict(path_to_json: Path) -> dict:
    with open(path_to_json, "r") as f:
        airtable_mapping_dict = json.load(f)
    return airtable_mapping_dict


def airtable_to_dataframe(
    airtable_api: Api, table_id: str, base_id: str
) -> pd.DataFrame:
    records = airtable_api.table(base_id, table_id).all()
    data_df = pd.DataFrame.from_records([r["fields"] for r in records])
    return data_df


@app.post("/add_review/")
def add_review_to_airtable(review: Review):
    # load airtable mapping dict:
    path_to_airtable_keys = "./airtable_key_mapping.json"
    airtable_ids_dict = load_airtable_mapping_dict(path_to_airtable_keys)

    # connect to airtable API:
    api = create_airtable_api("./token.txt")

    # create airtable table connections:
    base_id = airtable_ids_dict["base_id"]
    to_classify_reviews_id = airtable_ids_dict["reviews_to_classify_df"]
    to_classify_reviews_df = api.table(base_id, to_classify_reviews_id)
    try:
        to_classify_reviews_df.create(fields=review.dict())
        return {"status": "success"}
    except Exception as e:
        print(e)
        return {"status": "failed", "error_msg": print(e)}
