# Finetune SBERT models

This folder contains code to fine-tune an SBERT model on either Scifact or FEVER.

## How to use

### Create data
Call the following commands to create the data (it will create both SciFact and FEVER at the same time):
```python create_data.py <hugginface_model> <path_to_corpus_file> <path_to_scifact_train_file> <path_to_scifact_validation_file> <path_to_fever_train_file> <output_path>```

### Run finetuning script
```python finetune_sbert.py <hugginface_model> <data_folder_path> <number_of_fever_epochs> <number_of_scifact_epochs> -d {set if you want dense layer after pooling}```

Note: 
- the <data_folder_path> is the output_path given to create_data.py.
- You can set either <number_of_fever_epochs> or <number_of_scifact_epochs> to 0, if you only want to finetune on one of the datasets.