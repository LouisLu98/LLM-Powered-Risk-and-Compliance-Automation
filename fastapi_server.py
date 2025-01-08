from fastapi import FastAPI, File, UploadFile
import pandas as pd
import uvicorn
from data_process_model import predict, process_data, soc_violation_detetcion
import json
from pyngrok import ngrok, conf

app = FastAPI()

@app.get('/')
def index():
    return "Server is running...."

@app.post('/uploadfile/')
async def create_upload_file(file: UploadFile):
    # save to the tmp location
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # predict
    # For Category is Execute, area is pipeline or release
    #result = predict(file_location, "/content/drive/MyDrive/API/grid_search_model.joblib")
    result = predict(file_location, "grid_search_model.joblib")
    return {"filename": file.filename, "predictions": json.dumps(result)}


if __name__ == '__main__':
    # local
    #uvicorn.run('fastapi_server:app', host='127.0.0.1', port=8000, reload=True)

    # public
    ngrok.set_auth_token("2hMIaxG9ApIMY7e0SdAYrgcTqzn_7YWEjMSLnSesoYquHTLFo")

    # Start ngrok tunnel
    public_url = ngrok.connect(8000).public_url
    print(f"Public URL: {public_url}")

    # Run the app
    uvicorn.run('fastapi_server:app', host='127.0.0.1', port=8000)

#Swagger
#http://127.0.0.1:8000/docs
# Redoc
#http://127.0.0.1:8000/redoc
#ngrok
#ngrok config add-authtoken 2hMIaxG9ApIMY7e0SdAYrgcTqzn_7YWEjMSLnSesoYquHTLFo
#ngrok http 127.0.0.1:8000