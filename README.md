# fraud-detector
Fraud detector is a package for detecting fraud in transactions.

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


## TODOs and feature release backlog
- Feed business knowledge into model e.g. what is the cost to the business of False Negative / False Positives 
- Model training and optimisation, considering more extensive feature engineering and broader range of classifiers (e.g.: catboost) 
- Enhance feature selection, e.g.: look at feature importance and prioritise ordering used for variable selection
- Low level function and code documentation
- Add requirements.txt
