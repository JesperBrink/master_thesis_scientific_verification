# datasets

Fodler to hvae evrything with datasets.
Currently it has the scifact dataset, and scripts for creating the tfrecord datasets for both tarin and validation. 

To download the fever_datasets go to [drive](https://drive.google.com/drive/folders/1zzeXFBbpHaXpWl8dAoHpX7mWTl8Sayhq?usp=sharing) and download the desired fever_set.

To create the tfRecord data run:
`python createDataset.py {claim-dataset} {corpus} {relevant, notrelevant} {train, validation} -k {the number of not relevant sentences pr claim}`

Or download from: [drive](https://drive.google.com/drive/folders/1EtoDFtqIVKj0XBscWueBN5Ks64BfVVYG?usp=sharing).