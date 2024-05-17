import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import itertools

def le(df):
    """
    Apply label encoding to object-type columns in the DataFrame.

    Parameters:
        df (DataFrame): Input DataFrame.

    Returns:
        None
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])






def preprocess_data(train, test):
    """
    Preprocess the train and test data for machine learning.

    Parameters:
        train (DataFrame): Train dataset.
        test (DataFrame): Test dataset.

    Returns:
        X_train (array-like): Processed features for training.
        X_test (array-like): Processed features for testing.
        Y_train (array-like): Target variable for training.
        label_encoders (dict): Fitted label encoders.
    """
    le(train)
    le(test)

    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)

    # Feature selection
    X_train = train.drop(['class'], axis=1)
    Y_train = train['class']
    rfc = RandomForestClassifier()
    rfe = RFE(rfc, n_features_to_select=10)
    rfe = rfe.fit(X_train, Y_train)

    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
    selected_features = [v for i, v in feature_map if i==True]

    X_train = X_train[selected_features]
    X_test = test[selected_features]

    # Scaling
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)

    return X_train, X_test, Y_train, le








def process_data(data):
    """
    Preprocess the data for inference.

    Parameters:
        data (DataFrame): Input data.

    Returns:
        X (array-like): Processed features.
    """
    le(data)

    selected_features=['protocol_type', 'service',  'flag',  'src_bytes',  'dst_bytes',  'count',  'same_srv_rate',  'diff_srv_rate',  'dst_host_srv_count',  'dst_host_same_srv_rate']
    X = data[selected_features]
    scale = StandardScaler()
    X = scale.fit_transform(X)

    return X









from sklearn.preprocessing import LabelEncoder
import pandas as pd

def fit_label_encoder(df):
    """
    Fit a label encoder on each column with object dtype in the DataFrame.
    
    Parameters:
        df (DataFrame): Input DataFrame
        
    Returns:
        label_encoder (dict): Dictionary containing fitted label encoders for each column
    """
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            label_encoder.fit(df[col])
            label_encoders[col] = label_encoder
    return label_encoders



def transform_single_row(single_row, label_encoders):
    """
    Transform a single row of data using the fitted label encoders.
    
    Parameters:
        single_row (dict): Dictionary containing data for a single row
        label_encoders (dict): Dictionary containing fitted label encoders for each column
        
    Returns:
        encoded_single_row (dict): Dictionary containing encoded values for the single row
    """
    encoded_single_row = {}
    for col, value in single_row.items():
        if col in label_encoders:
            encoded_value = label_encoders[col].transform([value])[0]
            encoded_single_row[col] = encoded_value
        else:
            # If the value is not found in the fitted label encoder, handle it as needed
            encoded_single_row[col] = value
    return encoded_single_row