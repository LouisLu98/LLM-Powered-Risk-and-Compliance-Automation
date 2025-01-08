import requests
import pandas as pd
import time
# get new log data
#file_path = 'log_data_request.csv'
import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    # Not in Colab environment, proceed without mounting
    pass
file_path = 'content/drive/MyDrive/API/log_data_request.csv'

try:
# check api connection
    #url_test = requests.get('http://127.0.0.1:8000')
    url_test = requests.get('https://27e7-35-240-144-196.ngrok-free.app/')

    if url_test.status_code == 200:
        print(f"Connection successful")
    else:
        print(f"Failed to connection. Status code: {url_test.status_code}")
    # Send file to API
    with open(file_path, 'rb') as f:
        #response = requests.post('http://127.0.0.1:8000/uploadfile/', files={'file': f})
        response = requests.post('https://27e7-35-240-144-196.ngrok-free.app/uploadfile/', files={'file': f})
        print(f"Response from server: {response.json()}")
except Exception as e:
    print(f"An error occurred: {e}")

