# Usage guide
To join the battle to fraud, simply follow the following steps:
- 'git clone' the repo
- cd fraud-detector/bin
- Choose between one of the scripts below. For more options, check out the home [README](../README.md).

```
fraud-detector
└───bin
|   │  run_model_trainer.sh:        Script for offline model training
|   │  run_stream_scoring.sh:       Streaming applications which scans for incoming transactions from a target folder, 
|   │                               and generates predictions based on specified model based on a api. 
```