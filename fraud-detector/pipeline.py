

import conf.conf as conf
from utils import load_data_local
from feateng import FeatPipeline

def main():


    ###############################
    # 0. Variable Assignment
    ###############################

    inpath = conf.inpath
    intype = conf.intype

    ###############################
    # 1. Data read
    ###############################
    df = load_data_local(inpath, intype)

    ###############################
    # 2. Feature engineering pipeline
    ###############################
    df2 = FeatPipeline.fit_transform(df)


    ###############################
    # 3. ML pipeline
    ###############################



if __name__ == "__main__":
    main()