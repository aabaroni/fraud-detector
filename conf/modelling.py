######################################################################
# Configuration for fraud-detector/scoring_pipeline.py and predict_automation in api/app.py
######################################################################

inpath = "../data/input_data.csv"
intype = "csv"
sample_rate = 0.1

target_feature = ["isFraud"]
id_features = ['nameDest', 'nameOrig']
numeric_features = ['amount', 'newbalanceDest',
       'newbalanceOrig', 'oldbalanceDest', 'oldbalanceOrg', 'step', ]
categorical_features = ['stepDay', 'stepHour', 'type']
binary_features = ['isDestMerchant']


