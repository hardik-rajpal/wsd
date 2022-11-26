from fastapi import FastAPI
from fastapi.responses import JSONResponse,Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# from pydantic.typing import DictAny
from wsd import WSD

class Request(BaseModel):
    sentence:str#spaces replaced by _
    algorithm:str#wfs|mfs|elesk|pr

app = FastAPI()
w = WSD(True)

@app.post("/senses/")
async def create_item(req:Request):
    print(req)
    sent = ' '.join(req.sentence.split('_'))
    sensedict,taggedsent = w.attachSensesTo(sent,req.algorithm)
    sensedict = w.expandSenseDict(sensedict)
    return JSONResponse(content={
            'senses':sensedict,
            'tagged':taggedsent
            },headers={
                'Access-Control-Allow-Origin':'*'
            })

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# @app.options('/senses/')
# async def check():
#     return JResponse(,headers={
#         'Access-Control-Allow-Origin':'*'
#     })