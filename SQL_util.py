############# SQL database to export model information and loads the model information #################
########################################################################################################
from numpy import record
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.url import URL
import pandas as pd
from sqlalchemy.dialects.postgresql import *

def set_engine(conf):
    """Crete SQL Export & Import engine""" 
    return create_engine(URL(**conf), connect_args={'sslmode':'require'})


def delete_anomaly(table_name, name, type, creator,
                                  record_timestamp, gid_id, engine):
    if inspect(engine).has_table(table_name):
        #print()
        s = f"""Delete From live_data_anomaly Where model_creator = '{creator}' and 
                model_name = '{name}' and model_type = '{type}'and flow_meter = {gid_id} and flow_record_date = '{record_timestamp}' """
        with engine.begin() as conn:
            conn.execute(s)
# export table
def sql_table_export(table_name, name, creator, type, anomaly, label, 
                                  record_timestamp, gid_id,engine, date_update, flow):
    if inspect(engine).has_table(table_name):
        with engine.begin() as conn:
            conn.execute(f"""Delete From live_data_anomaly Where model_creator = '{creator}' and \
                model_name = '{name}' and flow_meter = {gid_id} and model_type ='{type}' and flow_record_date = '{record_timestamp}' """ )

    # Upload to database
    with engine.connect() as conn:
        print("***Uploading ready***")

        conn.execute("""Insert into live_data_anomaly (model_name,model_type, model_creator, anomaly_detection, anomaly_type, flow_record_date, flow_meter, date_updated, flow) 
        values(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",(name, type, creator, anomaly, label, record_timestamp, gid_id, date_update, flow))

        '''print("*** SQL START UPLOADING")

    print("*** SQL UPLOAD COMPLETED ***")'''
        
    return 

def get_model(model_name, creator, gid, engine):
    with engine.connect() as conn:
        model_string = conn.execute("""select model from anomaly_models where model_name = %s and model_creator = %s and flow_meter = %s""", 
        (model_name, creator, gid,)).fetchall()
    
    if len(model_string) == 0:
        raise Exception("Requested segment table is empty - %s" %(model_name))

    return model_string[0][0]

def get_scaler(model_name, creator, gid, engine):
    with engine.connect() as conn:
        model_string = conn.execute("""select scaler from anomaly_models where model_name = %s and model_creator = %s and flow_meter = %s""", 
        (model_name, creator, gid,)).fetchall()
    
    if len(model_string) == 0:
        raise Exception("Requested segment table is empty - %s" %(model_name))

    return model_string[0][0]

def get_errors(model_name, creator, gid, engine):
    with engine.connect() as conn:
        model_string = conn.execute("""select errors from anomaly_models where model_name = %s and model_creator = %s and flow_meter = %s""", 
        (model_name, creator, gid,)).fetchall()
    
    if len(model_string) == 0:
        raise Exception("Requested segment table is empty - %s" %(model_name))

    return model_string[0][0]

def get_table(data_name, gid, engine, gid_name='flow_gid', \
              datetime_name='datetime'):
    """
    Import flow data table from database 
    """
           
    # Required PostgreSQL query
    query = """ SELECT * FROM %s where %s = %s order by %s 
    """ %(data_name, gid_name, gid, datetime_name)
    # Import into dataframe
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    if len(df.index) == 0:
        raise Exception("Requested segment table is empty - %s" %(data_name))
        
    return df