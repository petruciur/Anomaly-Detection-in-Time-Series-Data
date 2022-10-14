"""
Functions to pickle the models
----------
The pickle tf model function is found from [https://github.com/tensorflow/tensorflow/issues/34697]
"""

import pickle as cPickle

import pickle

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def sql_table_export(data, c_id, m_name, comment,table_name, r2 ,engine):
    """
    Standardize export Models to cost_anlaysis interface
    """
    # Safety copy
        
    # Delete existing city values
    if not engine.dialect.has_table(engine, table_name):
        print("Target table does not exists  --  Will be created")

    else:
        with engine.begin() as conn:
            conn.execute("""Delete From cost_models Where city_id = %s and model_name = %s
            """ , (c_id, m_name,))

    # Upload to database
    with engine.connect() as conn:
        print("***Uploading ready***")
        conn.execute("""Insert Into cost_models (city_id, model_name, model_comments, models, r2_value) 
        values(%s,%s,%s,%s,%s)""",(c_id, m_name, comment, data, r2,))
        print("*** SQL START UPLOADING")

    print("*** SQL UPLOAD COMPLETED ***")
        
    return 


"""
For Fbprophet model
"""

def dump_regular_models (model):

    pkl_path = "/Users/jialingcai/Documents/GitHub/flow_models/Prophet.pkl"

    with open(pkl_path, "wb") as f:
        cPickle.dump(model, f)


    # model_dump = cPickle.dumps(model)
    # print("***Dumping the model***")

    return 

"""
For TF model
"""
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

# fix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


