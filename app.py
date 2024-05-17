import streamlit as st
from animations import load
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from data_pre_processing import process_data,le
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)



st.set_page_config(layout="wide")

if "result" not in st.session_state:
    st.session_state.result=None

def combine_dataframe(df0,df1):
    # Get the prediction dataframe from the session state
    df1 = pd.DataFrame(df1, columns=["Predicted results"])
    df0= pd.DataFrame(df0)
    df1['Predicted results'] = df1['Predicted results'].astype(object)

    df1.loc[df1['Predicted results'] == 0, "Predicted results"] = 'normal'
    df1.loc[df1['Predicted results'] == 1, "Predicted results"] = 'attack'



    # Add the original prediction dataframe as a new column
    combined_df = pd.concat([df1,df1])
    

    return df0


def create_result_dataframe(prediction_df,df1):
    # Get the prediction dataframe from the session state
    
    
    # Perform model fitting and prediction
    result_from_users = st.session_state.model.predict(prediction_df)
    if result_from_users == 0:
        result_from_users = ["normal"]
    else:
        result_from_users= ["attack"]

    # Create a new DataFrame with the prediction results
    result_df = pd.DataFrame({'Prediction Results': result_from_users},index=range(1))

    # Add the original prediction dataframe as a new column
    combined_df = pd.concat([df1,result_df], axis=1)

    return combined_df




if "df" not in st.session_state:
    st.session_state.df=None




st.title("Intrusion detection in a network")

csv,manual=st.tabs(["Upload a csv","Enter values normally"])
with csv:
    st.file_uploader("Upload a csv containing the apporpriate information",type=["csv"],key="file")

    attack = "animations/attack.json"
    normal ="animations/normal.json"

    with open('models/Naive_Baye.pkl', 'rb') as r:
        BNB_model = pickle.load(r)

    with open('models/RandomForestClassifier.pkl', 'rb') as s:
        RFC_model = pickle.load(s)

    with open("models/KNN_model.pkl", "rb") as a:
        KNN_model= pickle.load(a)

    with open("models/DecisionTreeClassifier.pkl","rb") as m:
        DTC_model= pickle.load(m)

if "model" not in st.session_state:
    st.session_state.model=BNB_model


if st.session_state.file:
    st.info("Value : 1= anomaly, 0= normal",icon="ℹ️")
    col1, col2, = st.columns([0.99,0.01])

    s= pd.read_csv(st.session_state.file)
    st.write("ssssssssssss")
    X_train = process_data(s)
    with col1:
        st.write(s)
    with col2:
        Y_train =BNB_model.predict(X_train)
        
        st.write(combine_dataframe(X_train,Y_train))
        st.write(Y_train)


def model_change():
    
    if st.session_state.model_select ==  "KNN Model (K-Nearest Neighbors)":
        st.session_state.model= KNN_model
    elif st.session_state.model_select == "Decision Tree classifier":
        st.session_state.model= DTC_model
    elif st.session_state.model_select=="Naive Bayes model":
        st.session_state.model= BNB_model
    elif st.session_state.model_select =="Random Forest Classifier":
        st.session_state.model= RFC_model
    else:
        st.session_state.model= BNB_model


with manual:
    selected_features = [  'src_bytes',  'dst_bytes',  'count',  'same_srv_rate',  'diff_srv_rate',  'dst_host_srv_count',  'dst_host_same_srv_rate']

    for opt in selected_features:
        st.number_input(f" Enter {opt}",key=opt, min_value=0)

    st.selectbox("Enter Protocol type",options=["tcp","udp","icmp"],index=None,key="protocol_type")

    st.selectbox("Enter Protocol type",options=["SF","S0","REJ","OTH","RSTR"],index=None,key="flag")

    st.selectbox("Enter Value for service",options=["ftp_data","other","private","http","remote_job","name","netbios_ns","eco_i","mtp","telnet","finger","domain_u","supdup","uucp_path","Z39_50","smtp","auth","netbios_dgm","csnet_ns","bgp","ecr_i","gopher","vmnet","systat","http_443","efs","imap4","whois","iso_tsap"],index=None,key="service")

    if st.session_state.protocol_type and st.session_state.flag and st.session_state.service:        
        
        data = {'protocol_type': [st.session_state.protocol_type],'service': [st.session_state.service], 'flag': [st.session_state.flag], 'src_bytes':[st.session_state.src_bytes], 'dst_bytes': [st.session_state.dst_bytes], 'count': [st.session_state.count], 
        'same_srv_rate':[st.session_state.same_srv_rate], 'diff_srv_rate': [st.session_state.diff_srv_rate], 'dst_host_srv_count': [st.session_state.dst_host_srv_count], 'dst_host_same_srv_rate': [st.session_state.dst_host_same_srv_rate]}
        st.session_state.df=pd.DataFrame(data,index=range(1))
        st.session_state.pf= pd.DataFrame(data,index=range(1))

        st.write(st.session_state.df)
        st.selectbox("Selec a model to use",options=["KNN Model (K-Nearest Neighbors)","Decision Tree classifier","Naive Bayes model"],on_change=model_change,key="model_select")
        
        def predict():
            le(st.session_state.df)
            st.session_state.result = create_result_dataframe(st.session_state.df,st.session_state.pf)
            value = st.session_state.result["Prediction Results"][0]
            if value== "normal":

                load(normal)
            else:
                load(attack)
            
            st.write(st.session_state.result)
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for a network analysis company you read network datasets and interpret your findings to people who might not understand it properly, the dataset that you analyze will always be related to network traffic analysis. The predicted result will always be between 'normal'- meaning the network is safe and 'attack' meaning there is something suspicious going on in the network communicate this to smartly list and explain briefly what all the parameters in the dataset provided mean and thier impact on the network. You should also recommend appropriate security measures if the network is under an attack to help beginners see if they can solve the attack "},
                {"role": "user", "content": f"{st.session_state.result}"},

            ]
            )

            st.write(response.choices[0].message.content)
        
            


        if st.button("make a prediction",type="primary"):
            predict()
 
            

    
      
        
        


