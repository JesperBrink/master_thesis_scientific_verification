from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from load_data import load_training_data, load_evaluator

model_save_path = "finetuned-stsb-distilbert-base"
scifact_claims_path = "../datasets/scifact/claims_sub_train.jsonl"
scifact_val_claims_path = "../datasets/scifact/claims_validation.jsonl"
scifact_corpus_path = "../datasets/scifact/corpus.jsonl"
fever_claims_path = "../datasets/fever/fever_train.jsonl"


model = SentenceTransformer("stsb-distilbert-base")

scifact_training_data, fever_training_data = load_training_data(scifact_claims_path, scifact_corpus_path, fever_claims_path)
scifact_train_dataloader = DataLoader(scifact_training_data, shuffle=True, batch_size=32)
fever_train_dataloader = DataLoader(fever_training_data, shuffle=True, batch_size=32)

train_loss = losses.CosineSimilarityLoss(model)
evaluator = load_evaluator(scifact_val_claims_path, scifact_corpus_path)

model.fit(
    train_objectives=[(fever_train_dataloader, train_loss)],
    epochs=3,
    evaluator=evaluator,
    evaluation_steps=500,
    output_path=model_save_path + "-fever",
)

scifact_model = SentenceTransformer(model_save_path + "-fever")

scifact_model.fit(
    train_objectives=[(scifact_train_dataloader, train_loss)],
    epochs=0,
    evaluator=evaluator,
    evaluation_steps=500,
    output_path=model_save_path + "-fever-scifact",
)