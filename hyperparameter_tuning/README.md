# Hyperparameter tuning

This module is used for doing hyperparameter tuning. 

The folder `hyperparameter_dicts` contains `sentence_selection_hyperparameter_dict.json` and `stance_prediction_hyperparameter_dict.json` which are the files used to define which hyperparameters should be tested when running `run_hyperparameter_tuning.py`.

## How to run the script
```
python run_hyperparameter_tuning.py <problem_type> <claims_path> <corpus_path>
```

- <problem_type>: sentence or stance
- <claims_path>: Path to jsonl file with embedded claim 
- <corpus_path>: Path to jsonl corpus file with embedded abstracts

Training data: the models are trained on the training data provided in the `subTrainingDataset` which is a subset of 90% of the full training data.