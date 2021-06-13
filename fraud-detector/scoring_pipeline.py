######################################################################
# Scoring pipeline, not used directly but inspired api/app.py
######################################################################


import conf.modelling as conf
from utils import load_data_local
from feateng import FeatPipeline
import joblib
from conf.modelling import target_feature, id_features

def main():


    ###############################
    # 0. Variable Assignment
    ###############################

    inpath = conf.inpath # input path of data to be scored
    intype = conf.intype # input path type of data to be scored
    model_chosen = conf.model_chosen

    ###############################
    # 1. Data read
    ###############################
    df = load_data_local(inpath, intype)
    all_features = df.columns.difference(target_feature).difference(id_features)

    ###############################
    # 2. Feature engineering pipeline
    ###############################
    df2 = FeatPipeline.fit_transform(df)

    ###############################
    # 3. Scoring
    ###############################
    model = joblib.load(model_chosen) + ".pkl"
    predictions = model.predict(df[all_features])
    prediction_prob = model.predict_proba(df[all_features])[:, 1]

    return predictions, prediction_prob



if __name__ == "__main__":
    main()