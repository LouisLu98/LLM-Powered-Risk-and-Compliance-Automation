import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
import joblib
from datasets import Dataset
from langchain import PromptTemplate,  LLMChain
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
import accelerate
import chromadb
from chromadb.utils import embedding_functions
import warnings
warnings.filterwarnings('ignore')

def process_data(file_path):
    import os
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        # Not in Colab environment, proceed without mounting
        pass
    datadf1 = pd.read_csv('/content/drive/MyDrive/API/data_combined_v20240601.csv')
    datadf2 = pd.read_csv(file_path)
    if 'Source' not in datadf1.columns or 'Source' not in datadf2.columns:
        datadf1['Source'] = 'log_data'
        datadf2['Source'] = 'process_data'

    datadf = pd.concat([datadf1, datadf2], ignore_index=True)
    # Cleaning the data
    datadf.dropna(subset=['Type'], inplace = True) #Dropping rows which are not AzureDevOpsAuditing Data
    datadf.drop_duplicates(inplace = True) #Dropping duplicates
    datadf['TimeGenerated [UTC]'] = pd.to_datetime(datadf['TimeGenerated [UTC]'])
    # Extracting logs from execute category only
    df = datadf[datadf['Category'] == 'Execute']
    df.reset_index(drop=True, inplace=True)


    # Extracting JSON data from Data column

    json_dict = df['Data'].apply(json.loads)

    json_df = pd.json_normalize(json_dict)

    for col in json_df.columns:
        json_df.rename({col: 'Data_' + col}, inplace=True, axis=1)

    data = pd.concat([df, json_df], axis=1)
    data.head()
    data = data[(data['Area'] == 'Pipelines') | (data['Area'] == 'Release')]

    col_drops = ['Data_CheckRuns','Data_ConnectionType','Data_ConnectionId','Data_DefinitionId','Data_OwnerDetails','Data_OwnerId','Data_PlanType','Data_ConnectionName','Data_CheckSuiteId','Data_CheckSuiteStatus']
    # Dropping columns with all NaN values
    for i in col_drops:
        try:
            data.drop(columns=[i], inplace = True)
        except:
            pass
    ## The missing values in the columns is associated with the deployment result
    cols_nan = ['AuthenticationMechanism','IpAddress','UserAgent','Data_ReleaseName','ProjectName','Data_PipelineId','Data_RequesterId','Data_RunName','Data_EnvironmentId','Data_AnonymousAccess','Data_FinishTime','Data_Result','Data_StartTime','Data_RequestId','Data_EnvironmentName','Data_JobName','ActorClientId']

    for col in cols_nan:
        try:
            data[col].fillna('Missing', inplace = True)
        except:
            pass


    ## Feature Selection & Feature Engineering

    # Remove columns with only one unique value
    cols_to_remove = []
    for col in data.columns:
        if data[col].nunique() == 1:
            cols_to_remove.append(col)

    print(f"Columns with only one unique value: {cols_to_remove}")
    # Remove columns with only one unique value
    data.drop(columns = cols_to_remove, inplace = True)

    # Dropping Data_StartTime column
    try:
        data.drop(columns = ['Data_StartTime'], inplace = True)
    except:
        pass

    # Dropping file_type column
    try:
        data.drop(columns = ['file_type'], inplace = True)
    except:
        pass

    ## Columns to drop:
    cols_to_drop = ['ProjectId', 'ScopeId', 'TenantId', 'Data_Result', 'ActorUserId', 'Data_AnonymousAccess', 'Data_RunName', 'Data_EnvironmentId', 'Data_PipelineId', 'ActorClientId','UserAgent', 'AuthenticationMechanism', 'IpAddress', 'Data_RequesterId','ActorUPN', 'ActorCUID','OperationName','Data_RequestId']
    for i in cols_to_drop:
        try:
            data.drop(columns = [i], inplace = True)
        except:
            pass
    try:
        data['ActorDisplayName'] = data['ActorDisplayName'].apply(lambda x: 1 if x == 'Azure DevOps Service' else 0)
    except:
        pass

    # Columns for frequency encoding:
    cols_for_freq_encoding = ['Data_PipelineName', 'Data_ReleaseName', 'Data_StageName','Data_EnvironmentName','Data_JobName']

    freq_encoding_maps = {}
    for col in cols_for_freq_encoding:
        try:
            freq_encoding = data[col].value_counts() / len(data)
            freq_encoding_maps[col] = freq_encoding
            data[col + '_encoded'] = data[col].map(freq_encoding)
        except:
            pass
    # Columns for one-hot encoding:
    cols_for_one_hot_encoding = ['ProjectName', 'Area','ScopeDisplayName','Data_CallerProcedure']
    # Filter the columns to only include those that exist in the dataset
    existing_cols_for_one_hot_encoding = [col for col in cols_for_one_hot_encoding if col in data.columns]

    # Perform one-hot encoding only on the existing columns
    data = pd.get_dummies(data, columns=existing_cols_for_one_hot_encoding)

    # Removing all columns not needed in the model and the target variable

    target = data['Data_DeploymentResult']
    cols_to_remove = ['CorrelationId','ActivityId','TimeGenerated [UTC]','Data','Details','Data_FinishTime','Data_DeploymentResult', 'Date']
    for i in cols_to_remove:
        try:
            data.drop(columns = [i], inplace = True)
        except:
            pass
    # Removing columns which were frequency encoded
    for i in cols_for_freq_encoding:
        try:
            data.drop(columns = [i], inplace = True)
        except:
            pass
    # Label encoding the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target)
    x = data
    processed_log_data = data[data['Source'] == 'process_data']
    log_data_ids = processed_log_data['Id']
    col_remove = ['Source', 'Unnamed: 27','label','Id']
    for i in col_remove:
        try:
            processed_log_data.drop(columns = [i], inplace = True)
        except:
            pass

    print(processed_log_data.columns)
    print(processed_log_data)
    print(processed_log_data.iloc[0])
    return log_data_ids, processed_log_data


def soc_violation_detetcion(file_path):
    import pandas as pd
    import numpy as np
    import json
    from sklearn.preprocessing import LabelEncoder
    import joblib
    from datasets import Dataset
    from langchain import PromptTemplate, LLMChain
    from langchain import HuggingFacePipeline
    from transformers import AutoTokenizer, pipeline
    import torch
    import accelerate
    import chromadb
    from chromadb.utils import embedding_functions

    ## Load data
    df = pd.read_csv(file_path)
    df = df[df['Category'] == 'Execute']

    def clean_text(text):
        cleaned_text = text.replace('"', '').lower()
        return cleaned_text

    # Filter out rows that contain "success", "succeed", or variations
    filtered_data = df[~df['Details'].str.contains('success|succeed', case=False)]

    # Clean the text data
    filtered_data['cleaned_text'] = filtered_data['Details'].apply(clean_text)
    dataset = Dataset.from_pandas(filtered_data)

    print(filtered_data)
    ids = filtered_data['Id']

    ## Initial model
    from langchain import HuggingFacePipeline
    from transformers import AutoTokenizer, pipeline
    import torch
    import accelerate
    # from langchain import HuggingFacePipeline
    from transformers import AutoModelForCausalLM
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(name="test1")
    except:
        pass
    collection = chroma_client.create_collection(name="test1")
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    model = "tiiuae/falcon-7b-instruct"  # tiiuae/falcon-40b-instruct

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        max_new_tokens=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})
    template = """
    You are an intelligent chatbot. Help the following question with brilliant answers.
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    def create_question(example):
        example['question'] = f"What SOC standard does this operation violate? '{example['cleaned_text']}'"
        return example

    # Apply the function to create questions
    dataset = dataset.map(create_question)

    def truncate_text(text, max_length):
        # Ensure the input text is a string
        if isinstance(text, str):
            tokenized_text = tokenizer.encode(text, truncation=True, max_length=max_length)
            return tokenizer.decode(tokenized_text, skip_special_tokens=True)
        else:
            raise TypeError("Input text must be a string")

    def get_answer(examples):
        max_input_length = 250 - 50  # Set a buffer (e.g., 50 tokens)
        truncated_questions = [truncate_text(question, max_input_length) for question in examples['question']]
        examples['SOC_violation'] = [llm_chain.run(question) for question in truncated_questions]
        return examples

    dataset = Dataset.from_dict({"question": dataset['question'], "Details": dataset['Details']})

    results = dataset.map(get_answer, batched=True, batch_size=10)

    results_df = results.to_pandas()
    results_df['SOC_violation'] = results_df['SOC_violation'].str.extract(r'Answer:\s*(.*)')
    soc_violation_dict = {}
    for i in range(len(results_df)):
        soc_violation_dict[ids[i]] = results_df['SOC_violation'][i]

    return soc_violation_dict


def predict(file_path, model_path):
    log_data_ids, data = process_data(file_path)
    soc_violation_dict = soc_violation_detetcion(file_path)
    #soc_violation_dict = {
    #    '3c4bdc31-56bb-45db-8d9a-22ce554f0988': 'The violation of this SOC standard is due to the deployment being outside the scope of the configuration management system.',
    #    '893dc326-1297-4489-bc50-8201610ee1a2': 'The operation violates a SOC-standard for unauthorized access and data exfiltration. This is a security risk.'}
    model = joblib.load(model_path)
    predictions = model.predict(data)
    data['Predicted_Label'] = predictions
    mapping_dict = {
        0: 'Canceled',
        1: 'Failed',
        2: 'Not deployed',
        3: 'Succeeded',
        4: 'Succeeded with issues',
        5: 'partially succeeded'
    }
    result = data['Predicted_Label'].map(mapping_dict)
    print(f"log_data_ids", log_data_ids)
    print(f"result", result)
    print(f"result[0]", result.iloc[0])
    print(f"log_data_ids[i]", log_data_ids.iloc[0])
    result_dict = {}
    for i in range(len(log_data_ids)):
        if result.iloc[i] == 'Failed':
            result_dict[log_data_ids.iloc[i]] = [result.iloc[i], soc_violation_dict[log_data_ids.iloc[i]]]
        else:
            result_dict[log_data_ids.iloc[i]] = result.iloc[i]
    print(result_dict)
    return result_dict
