# Abstract Retrieval experiment
TODO: add sacret to this experiment

## How to run
TODO

### BioSentVec
If you want to use BioSentVec for abstract retrieval, you need the BioSentVec encoded claims and corpus. 

Easiest solution is to download them [here](https://drive.google.com/drive/folders/1XVRfqjKOCAQeQscsR7F8H3REn3pDLVdt?usp=sharing), where the following files can be found:  corpus_paragraph_biosentvec.pkl and claim_biosentvec_{train, dev, test}.pkl. These files was generated with the `ComputeBioSentVecAbstractEmbedding.py` script provided by the authors of the Paragraph-joint solution, and can be found [here](https://github.com/jacklxc/ParagraphJointModel).

Note: if this does not work for some reason, you can generate these files by running the ComputeBioSentVecAbstractsEmbedding script. However, it requires you to [download](https://github.com/ncbi-nlp/BioSentVec#biosentvec) the 21GB BioSentVec model.