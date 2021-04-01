from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from load_data import load_training_data, load_evaluator

model_save_path = "temp-model"
claims_path = "../datasets/scifact/claims_sub_train.jsonl"
corpus_path = "../datasets/scifact/corpus.jsonl"


model = SentenceTransformer("stsb-distilbert-base")

training_data = load_training_data(claims_path, corpus_path)
train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
#evaluator = load_evaluator()

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    #evaluator=evaluator,
    evaluation_steps=500,
    output_path=model_save_path,
)
