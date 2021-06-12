# Data
toscore_inpath = "../data/sample_for_scoring.csv"
toscore_intype = "csv"

# Model chosen - Fill after training the models
model_chosen = "../models/gradientBoosting_4"
features_used = ['amount', 'isDestBalanceNewZero', 'isDestBalanceOldZero','isDestMerchant']

# Output location
output_location_api_tagged = "../data/"

      # #, 'isFlaggedFraud', 'isOrigBalanceNewZero',
      # 'isOrigBalanceOldZero', 'newbalanceDest', 'newbalanceOrig',
      # 'oldbalanceDest', 'oldbalanceOrg', 'step', 'stepDay', 'stepHour',
      # 'stepWeekDay', 'type']