from fastapi import FastAPI
from api import sports_predict
import uvicorn

sports_app = FastAPI(title='Sports Classification')

sports_app.include_router(sports_predict.sports_predict_router)

if __name__ == '__main__':
    uvicorn.run(sports_app, host='127.0.0.1', port=8000)
