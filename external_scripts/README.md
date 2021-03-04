# External scripts

This folder consists of files taken from others. They are helper scripts and mapping scripts atm.

to Preprocess_fever.py:
1. Download Fever dataset:

    `wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl`
    
    `wget https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl`

    `wget https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl`

2. Download Wikipedia Dump and unzip it manually.

    `wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip`

3. Run:

    `python preprocess_fever.py {wiki-folder} {train, dev, test} {output-file}`