## Getting Started

First, run npm install:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
```

<<<<<<< HEAD
Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.


## Data

Takes as input a JSON lines (.jsonl) file, where each JSON object contains a result-object containing the results and their corresponding names, as well as an optional params-object containing the hyperparameters and their corresponding names.

### Example data

```json
{
    "params":{
        "class_weight_0":1,
        "class_weight_1":1,
        "dense_units":256,
        "fever_epochs":3,
        "scifact_epochs":5,
        "threshold":0.5
    },
    "results":{
        "Abstract Retrieval Precision":0.0004531597594133641,
        "Abstract Retrieval Recall":0.5892857142857143,
        "Abstract Retrieval F1":0.0009056230961332637,
        "Sentence Selection Precision":0.0002525275300101913,
        "Sentence Selection Recall":0.27722772277227725,
        "Sentence Selection F1":0.0005045954225986664
    }
}
```
=======
Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.
>>>>>>> 7573a4b1e387bb080f66c5532f0716c1d3a1c6b3
