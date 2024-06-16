from typing import Union
from fastapi import FastAPI, Query
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

from main_app import QnA_with_LLM

app = FastAPI()


class QueryModel(BaseModel):
    q: str

@app.post("/api/query")
async def read_query(query: QueryModel):
    res = QnA_with_LLM(query)
    return {"result": res}


"""
curl -X 'POST' \
  'http://127.0.0.1:8000/api/query' \
  -H 'Content-Type: application/json' \
  -d '{"q": "запрос"}'

"""  