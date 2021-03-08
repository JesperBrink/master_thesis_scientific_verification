# Hyperparameter tuning

This module is used for doing hyperparameter tuning. 

The folder `hyperparameter_dicts` contains `sentence_level_hyperparameter_dict.json` and `stance_prediction_hyperparameter_dict.json` which are the files used to define which hyperparameters should be tested when running `run_hyperparameter_tuning.py`.

## How to run the script
```
python run_hyperparameter_tuning.py <problem_type> <train_data_path> <claims_path> <corpus_path>
```

- <problem_type>: sentence or stance
- <train_data_path>: Path to folder with tfrecords training files
- <claims_path>: Path to jsonl file with embedded claim 
- <corpus_path>: Path to jsonl corpus file with embedded abstracts