CONDA_BASE=$(conda info --base)
source "$CONDA_BASE"/etc/profile.d/conda.sh

conda create -n scifact-env python=3.9 -y
conda activate scifact-env
conda install -y tensorflow-gpu=2.4.1 pandas jsonlines tqdm 

pip install torch==1.7.1 sentence-transformers sklearn pyserini rank_bm25
