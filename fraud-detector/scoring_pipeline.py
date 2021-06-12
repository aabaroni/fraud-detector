

import conf.conf as conf
from utils import load_data_local
from feateng import FeatPipeline
import joblib


def main():


    ###############################
    # 0. Variable Assignment
    ###############################

    inpath = conf.inpath # input path of data to be scored
    intype = conf.intype # input path type of data to be scored
    model_chosen = conf.model_chosen
    features_used = conf.features_used

    ###############################
    # 1. Data read
    ###############################
    df = load_data_local(inpath, intype)[features_used]

    ###############################
    # 2. Feature engineering pipeline
    ###############################
    df2 = FeatPipeline.fit_transform(df)

    ###############################
    # 3. Scoring
    ###############################
    model = joblib.load(model_chosen) + "pkl"
    #predictions = predict(df)
    #prediction_prob = predict_proba(df)[:, 1]



if __name__ == "__main__":
    main()