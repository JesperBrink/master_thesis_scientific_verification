# datasets

Fodler to hvae evrything with datasets.
Currently it has the scifact dataset, and scripts for creating the tfrecord datasets for both tarin and validation. 

To download the fever_datasets go to [drive](https://drive.google.com/drive/folders/1zzeXFBbpHaXpWl8dAoHpX7mWTl8Sayhq?usp=sharing) and download the desired fever_set.

To create the tfRecord data run:
`python basic/createDataset.py {claim-dataset} {train, validation} -f {if FEVER} -c {corpus_path, only used if not fever} -k {the number of not relevant sentences pr claim, only used of not fever} -r {relevant or notrelevant, only used if not fever}` 

Or download from: [drive](https://drive.google.com/drive/folders/1EtoDFtqIVKj0XBscWueBN5Ks64BfVVYG?usp=sharing).

In the lstm folder you can create the dataset needed to predict sentence selection with the lstm model.
To create the tfRecord data for the lstm model run:
`python lstm/createDataset.py {claim-dataset} {train, validation} -f {if FEVER} -c {corpus_path, only used if not fever} -k {the number of not relevant abstracts that will be used pr claim}`