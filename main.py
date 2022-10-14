
########### mian script ##########
from dataclasses import dataclass
from sqlite3 import Timestamp
import pandas as pd
import numpy as np
import argparse
import os
import pickle as cPickle
#import pickle
import warnings

from sklearn import linear_model
import ssl
import json
from datetime import datetime
from time import gmtime, strftime, time
from datetime import datetime, timedelta
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow
import tensorflow as tf
import warnings
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import SQL_util
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.keras.models import Sequential, Model

def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # noinspection PyPackageRequirements
        import tensorflow as tf
        from tensorflow.python.util import deprecation

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):  # pylint: disable=unused-argument
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        deprecation.deprecated = deprecated

    except ImportError:
        pass

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")

def get_table(data_name, gid, engine):
    """
    Import flow data table from database 
    """
           
    # Required PostgreSQL query
    query = """ SELECT * FROM %s where flow_gid = %s
    """ %(data_name, gid)

    # Import into dataframe
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if len(df.index) == 0:
        raise Exception("Requested segment table is empty - %s" %(data_name))
        
    return df

def create_parser():
    """
    Initialize argument parser
    """
    # Create parser
    parser = argparse.ArgumentParser()

    # Add city argument
    parser.add_argument("--gid", help="flow meter ID (gid) to get flow prediction", action="store")
    parser.add_argument("--model", help="model", action="store")
    parser.add_argument("--starttime", help="start stime", action="store")
    parser.add_argument("--endtime", help="end stime", action="store")
    parser.add_argument("--alpha", default=0.002,help="significannce level", action="store")


    return parser
    
if __name__ == "__main__":
    
    tensorflow_shutup()

    # Get project root
    root = os.path.join(os.getcwd()) 

    # Initialize parser
    parser = create_parser()

    # Parse command line argument
    args = parser.parse_args()
    gid = args.gid
    ts = args.starttime
    model_type = args.model
    ts_end = args.endtime
    alpha = float(args.alpha)

    # Convert script times into timestamps
    start = pd.Timestamp(ts)
    end = pd.Timestamp(ts_end)
    step = pd.Timedelta(hours=1,days=0)

    # Connect to the database
    json_conf = os.path.join(root, "config_database.json")
    with open(json_conf, "r", encoding="utf-8")  as file:
            sql_config = json.load(file)
    model_engine = SQL_util.set_engine(sql_config['model_db'])
    table_name = "live_data_anomaly"
    model_creator = 'Petru Ciur'
    date_update =  pd.to_datetime(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    gid_id = int(gid)

    # Get live data into a dataframe
    df = SQL_util.get_table("model_covariates", gid, model_engine)
    df.set_index('datetime', inplace=True)

    # Get fillna data (used to fill in anomalies or empty data).
    df_fillna = SQL_util.get_table("real_time_fillna_result", gid, model_engine)
    df_fillna.set_index('datetime', inplace=True)


    # LSTM-based models need special adjustments
    if model_type in ['lstm_alpha_quantile','lstm_alpha_prob']:

      model_name = "LSTM_err"
      scaler = cPickle.loads(\
                SQL_util.get_scaler(model_name, model_creator,gid, model_engine))

      # Standarize live data flow values
      df['flow'] = scaler.transform(df['flow'].values.reshape(-1,1))

      # Standarize fillna data flow values
      df_fillna['filled_flag_flow'] = scaler.transform(df_fillna['filled_flag_flow'].values.reshape(-1,1))
      
      # Augment dataframe with additional attributes
      k = 24
      # Flow of last k hours
      for i in range(1, k+1):
        df[f'flow_{i}'] = df.flow.shift(i)
      
      # Last hour's precipitation values
      df['precipitation_1'] = df.precipitation.shift(1)

      # Temperature of 1 and 4 hours ago
      df['temperature_1'] = df.temperature.shift(1)
      df['temperature_4'] = df.temperature.shift(4)

      # api30 values from one and 4 hours ago
      df['api_30_1'] = df.api30.shift(1)
      df['api_30_4'] = df.api30.shift(4)

      # average precipitation of 1 hour ago
      df['avg_precip_1'] = df.avg_precip.shift(1)

      # Disregard first k lines (they will contain NAN values)
      df = df[k:]

      X = df[[f'flow_{i}' for i in range(1,k+1)] + ['precipitation_1', 
                                                    "temperature_1", 'temperature_4',
                                                    'api_30_1', 'api_30_4', 'avg_precip_1']]

    # Initialize current timestamp 'ts' with the 'start time'
    ts = start
                                                   
    flows , anomalies, predictions = [], [], []
    while (ts <= end): # Iterating through each hour from 'start time' to 'end time'
        
        # We assume the flow at 'ts' is NOT an anomaly
        anomaly = 0

        # simplemath
        if model_type == 'simplemath':
            model_name = "SM_stdev_4"

            # We assume there is no precipitation
            precipitation = 0

            # If the current 'ts' is outside the time period on which live data is available.
            if  pd.to_datetime(ts) not in df.index:
                print(f"date {ts} out of available range!")

            else:
                # Get flow value measured at time 'ts'
                data = df.loc[pd.to_datetime(ts), 'flow']

                # If the 'ts' from one hour ago is outside the time period on which live data is available.
                if (pd.to_datetime(ts - step) in df.index) or (pd.to_datetime(ts - step) in df_fillna.index):
                    if pd.to_datetime(ts - step) in df.index:
                        data_prev = df.loc[pd.to_datetime(ts - step) , 'flow']
                    elif pd.to_datetime(ts - step) in df_fillna.index:
                        data_prev = df.loc[pd.to_datetime(ts - step) , 'flow']
                else:
                    anomaly = 1
                    print('previous data absent', ts)
                    label = model_type = "previous data absent"
                if anomaly != 1:
                    try:
                        # 'precipitation' = 1 if there is any rain in the last 4 hours
                        for i in range(4):
                            if df.loc[pd.to_datetime(ts - pd.Timedelta(hours=1,days=0)), 'precipitation'] > 0:
                                precipitation = 1 
                                break
                    except:
                        print('not enough info about precipitation, but it is fine!')

                # Get 'simplemath' model                
                model = cPickle.loads(\
                    SQL_util.get_model(model_name, 
                                        model_creator, 
                                        gid, 
                                        model_engine
                                        ))

                # Predict if 'data' is an anomaly
                (anomaly, label) = model.predict(data, data_prev, precipitation, hour = ts.hour)
                if anomaly:
                    print("label=",label)
                    print("flow=", data)

        elif model_type in ['lstm_alpha_quantile' ,'lstm_alpha_prob']:

            # If the current 'ts' is outside the time period on which live data is available.
            if not pd.to_datetime(ts) in df.index:
                print(f"WARNING! No available flow data on {ts}")

            elif np.isnan(df.loc[pd.to_datetime(ts), 'flow']):
                anomaly = 1
                label = 'nan value'
                data = -1
                print(f"NAN value at {ts}")

            else:
                model_name = "LSTM_err"

                # Get model
                model = cPickle.loads(\
                    SQL_util.get_model(model_name, model_creator,gid, model_engine))

                # Get errors list
                errors = cPickle.loads(\
                    SQL_util.get_errors(model_name, model_creator,gid, model_engine))

                # Select only the columns relevant for the prediction.
                X = df[[f'flow_{i}' for i in range(1,k+1)] + ['precipitation_1', \
                                                        "temperature_1", 'temperature_4', \
                                                        'api_30_1', 'api_30_4', 'avg_precip_1']]

                # Select the line corresponding to the current time 'ts'                                                         
                X = X[str(ts):str(ts)]

                # Fill NAN flow data.
                for i in range(1,24+1):

                    f = X[f'flow_{i}'].values[0]

                    # If any of the flows correposnding to the last 24 hours if NAN, replace fillna data.
                    if np.isnan(f) and pd.to_datetime(ts - pd.Timedelta(hours=i,days=0)) in df_fillna.index:
                        fill = df_fillna.loc[pd.to_datetime(ts - pd.Timedelta(hours=i,days=0)), 'filled_flag_flow']
                        X[f'flow_{i}'] = fill

                    # If the _unscaled_ flow is negative, replace with 0.
                    elif scaler.inverse_transform([[f]])[0][0] < 0:
                        X[f'flow_{i}'] = scaler.inverse_transform([[0]])[0][0]

                # SCALED flow
                data = df.loc[pd.to_datetime(ts), 'flow'] 
                
                # Reshape X for prediction.
                X = X.values.reshape(-1,30, 1)

                yhat_db = model.predict(X) # predicted (scaled) flow
                y_db = data # real (scaled) flow
                err_db = abs(yhat_db - y_db) # absolute error
                
                if model_type in ['lstm_alpha_quantile']:
                    # Get the quantile treshold
                    q = pd.DataFrame(errors).quantile(q=(1-alpha)).values[0] # 99% rejection region
                    
                    # set treshold
                    treshold = q

                    # label anomaly
                    label = 'extremely unlikely (q)'

                elif model_type in ['lstm_alpha_prob']:

                    # Get the probability treshold {p : P(X > p) = alpha}
                    p = (-1) * np.log(alpha) / (1/np.mean(pd.Series(errors).mean())) # 99% quantile

                    # set treshold
                    treshold = p

                    # label anomaly
                    label = 'extremely unlikely (p)'

                # anomaly = 1 (or true) if error exceeds qunatile treshold
                anomaly = int(err_db > treshold)
                #print(ts, anomaly)
        
                # data rescaled back to normal (since that's what we want to uplaod to DB)
                data = scaler.inverse_transform([[data]])[0][0] 

        # Delete anomaly detected by 'model_type' at time 'ts'.
        SQL_util.delete_anomaly(table_name, model_name, model_type, model_creator, ts, gid_id, model_engine)

        # In case that an anomaly has been detected this time by 'model_type' at time 'ts', it will be uploaded
        # in the following lines.

        # Save true value and prediction of flow at time 'ts'.

        # If anomaly was detected.
        if anomaly == 1:
            print("Anomaly date:", ts)
            if model_type in ['lstm_alpha_quantile','lstm_alpha_prob'] and data != -1:

                print("Real (not normalized) Flow Value :", data)
                print("Normalized Flow :", y_db)
                print("Normalized Flow Prediction : ", yhat_db[0][0])
                print('Normalized Prediction Error :',err_db[0][0])
                print('Normalized Error Treshold', treshold)

            
            # Boolean for wether there has been rain in the last 3 hours.
            last_hours_precip = df.loc[pd.to_datetime(ts - pd.Timedelta(hours=3)) : pd.to_datetime(ts), 'precipitation'].any()

            # Only upload anomalies which are not in rain periods.
            if not last_hours_precip:
                SQL_util.sql_table_export(table_name, model_name, model_creator, model_type, anomaly, label,
                                        ts, gid_id, model_engine, date_update, data)
                print("+++ Anomaly uploaded +++")
                print("========================")
            else:
                print("### Anomaly NOT uploaded due to recent rain period ###")
                print('===============')
        
        # Advance 'ts' with one hour.
        ts += step
        
    print("***Anomaly Detection Done!***")
