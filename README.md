# fraud-detector
Fraud detector is a package for detecting fraud in transactions.

## Business context 
As a financial institution, it is important that fraudulent transactions are blocked to avoid losses (fraud refunds etc.),but even more importantly to maintain the trust of customers and good reputation in the market.  
Based on the 1 month sample used, about 0.1% of transactions are marked as fraudulent, representing circa 1% of transaction value. 
> ### User stories  
>  - As a member of the fraud prevention team, I want a system that flags any suspicious transactions in real time
>  - The system needs to be scalable in production, handling an amount of transactions 20x bigger than the data provided, i.e. circa 4Mln daily transactions 


## Architecture
For current architecture and considerations on road to production, see [here](docs/architecture.md)

## Usage guide
See [here](docs/usage-guide.md)

## Project Structure
```
fraud-detector
│   README.md
│   requirements.txt:               Environment information    
└───api
|   │  app.py:                      fastapi app that creates predictions based on specified input data and classifier
|   │ 
└───automation
|   │  accumulator.py:              Accumulator scans for untagged transactions, and calls api_scorer to create predictions 
|   │  api_scorer.py:               Scores a list of input files using fastapi
|   │ 
└───bin
|   │  run_model_trainer.sh:        Script for offline model training
|   │  run_stream_scoring.sh:       Streaming applications which scans for incoming transactions from a target folder, 
|   │                               and generates predictions based on specified model based on a api. 
└───conf
|   │  api.py: Configuration for api/app.py
|   │  automation.py:               Configuration for automation/accumulator.py
|   │  modelling.py:                Configuration for fraud-detector/scoring_pipeline.py and predict_automation in api/app.py
└───data: dump of inputs/outputs data files. Default folders listed below but can be updated in conf/automation.py 
|   │  automation_in
|   │  automation_out
|   │ 
└───fraud-detector:                 Code for feature engineering, model training and scoring pipeline
|   │  feateng.py:                  Feature engineering pipeline
|   │  model_train_comparison.py:   Modelling pipeline for fraud detection, see file introduction for more details
|   │  scoring_pipeline.py:         Scoring pipeline, not used directly but inspired api/app.py
|   │
└───logs:                           Contains job logs
└───models:                         Stores models in pickle format 
|   │ 
└───notebooks:                      Mostly sandbox for code build and analysis. See eda.ipynb for Exploratory Data Analysis
```


#### TODOs and feature release backlog
- Feed business knowledge into model e.g. what is the cost to the business of False Negative / False Positives 
- Model training and optimisation, considering more extensive feature engineering and broader range of classifiers (e.g.: catboost) 
- Enhance feature selection, e.g.: look at feature importance and prioritise ordering used for variable selection
- Low level function and code documentation
- Add requirements.txt
