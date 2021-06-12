
inpath = "../data/input_data.csv"
intype = "csv"
sample_rate = 0.1
input_features = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]
id_features = ['nameDest', 'nameOrig']
numeric_features = ['amount', 'newbalanceDest',
       'newbalanceOrig', 'oldbalanceDest', 'oldbalanceOrg', 'step', ]
categorical_features = ['stepDay', 'stepHour', 'type']
binary_features = ['isDestMerchant']
target_feature = ["isFraud"]


