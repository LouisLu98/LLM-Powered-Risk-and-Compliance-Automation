
README
Overview
This project involves setting up a FastAPI server to handle predictions on log data. The data should be structured similarly to the training data. The process includes preparing a CSV file with the log data, running the FastAPI server, and using an API client to make predictions.

Prerequisites
Python 3.7+
Required Python packages:
fastapi
uvicorn
pydantic
pandas
scikit-learn
joblib

Step-by-Step Instructions
1. Prepare the CSV File
Before starting, ensure you have a CSV file containing the log data for prediction. The structure of this CSV file must match the structure of the training data used for building the model.ata structure exactly.

2. Run the FastAPI Server
	1. Start the Server: Run the FastAPI server 'fastapi_server.py'
	2. Test the Connection: Use the GET endpoint to test if the server is running successfully. If the connection is 		successful, you should see the message: "Server is running….".

3. Use the API Client
	1. Set Up the API Client: The API client script 'api_client.py' is used to make predictions. This script imports 'data_process_model.py', which automatically handles data processing to match the model's requirements.
	2.Call the Prediction Function: The prediction function in 'data_process_model.py' will be called from api_client.py. The function processes the data and outputs a dictionary with the structure:{log_data_unique_id: predict_result}.


Example Usage
1.Prepare the CSV file (example: log_data.csv).
2.Run the FastAPI server.
3.Check server status: Access the GET endpoint to verify the server is running.
4.Run the API client.

File Descriptions
fastapi_server.py: Sets up and runs the FastAPI server. This script also imports a pre-trained model saved using joblib.
api_client.py: Client script to send log data to the server and retrieve predictions.
data_process_model.py: Handles data processing and model prediction.


Notes
Ensure that the structure of the CSV file used for predictions matches the training data structure exactly.
The server should be running before using the API client to make predictions.
The pre-trained model should be saved and loaded using joblib in fastapi_server.py.