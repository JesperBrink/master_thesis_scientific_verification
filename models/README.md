# Models

this section is split into 4 parts abstract retrieval, sentence filter, sentence selection and statnce prediction 

## Abstract retrieval
An abstract retreiver takes a claim_object and a map with a doc_id to abstract mapping 
e.g.:
```python
Retriever(
    {"id": 1337, "claim":"hello my friend"},
    {
        "1": ["nope"],
        "12489688": ["hello 1", "hello 2"], 
        "13515165": ["hello 3", "hello 4", "hello 5"],
    }
)
```
It returns a subset of the map provided mapping from doc_id to the sentences in the abstract:
```python
{   
    "12489688": ["hello 1", "hello 2"], 
    "13515165": ["hello 3", "hello 4", "hello 5"],
}
```

## Sentence filter
TBD...

## Sentence selection
A sentence selector takes a claim_object, and a map with a doc_id to abstract mapping (could be output from abstract retireval section):
```python
Selector(
    {"id": 1337, "claim":"hello my friend"},
    {   
        "12489688": ["hello 1", "hello 2"], 
        "13515165": ["hello 3", "hello 4", "hello 5"],
    }
)
``` 
It returns a map with doc_id to a list of indices with the selected sentence for the given doc_id
```python
{
    "13515165": [3,0],
}
```

## Stance prediction
A Stance predictor takes a claim_object, a map from doc_id to selected sentences (output from sentence selector) and a doc_id to abstract map (result from abstract retriever)
```python
Predictor(
    {"id": 1337, "claim":"hello my friend"},
    {"13515165": [3,0]},
    {   
        "12489688": ["hello 1", "hello 2"], 
        "13515165": ["hello 3", "hello 4", "hello 5"],
    }
)
```
It returns the expected output to solve the sciver task, a map with claim_id and evidence, which is the result of the sentence selector and a label to those sentences
```python
{'id': 1337, 'evidence': {13515165: {'sentences': [3, 0], 'label': 'CONTRADICT'}}}
```