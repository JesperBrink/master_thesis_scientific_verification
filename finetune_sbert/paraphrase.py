import json
import logging
from pathlib import Path
import sys
import torch

from scipy import spatial
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

class Paraphraser():
    def __init__(self, embedding_model):
        self._set_seed(42)
        self.tokenizer = T5Tokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws')
        self.model = T5ForConditionalGeneration.from_pretrained('Vamsi/T5_Paraphrase_Paws')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.max_len = 256
        self.embedding_model = SentenceTransformer(embedding_model)

    def paraphrase_claim_and_sentences(self, claim, sentences):
        claim_and_sentence_pairs = []

        paraphrased_claim = self._paraphrase(claim)
        paraphrased_sentences = [self._paraphrase(sentence) for sentence in sentences]

        for sentence in sentences:
            claim_and_sentence_pairs.append((claim, sentence))
            if paraphrased_claim is not None:
                claim_and_sentence_pairs.append((paraphrased_claim, sentence))

        for paraphrased_sentence in paraphrased_sentences:
            if paraphrased_sentence is None:
                break
            claim_and_sentence_pairs.append((claim, paraphrased_sentence))
            if paraphrased_claim is not None:
                claim_and_sentence_pairs.append((paraphrased_claim, paraphrased_sentence))

        return claim_and_sentence_pairs

    def _paraphrase(self, sentence):
        text = "paraphrase: " + sentence
        
        encoding = self.tokenizer.encode_plus(text, padding=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding[
            "attention_mask"].to(self.device)

        paraphrased_sentences = self._generate_paraphrased_sentences(sentence, input_ids, attention_masks, 20)            
        least_similar_paraphrased_sentence = self._get_least_similar_paraphrased_sentence(sentence, paraphrased_sentences)
        return least_similar_paraphrased_sentence

    def _generate_paraphrased_sentences(self, sentence, input_ids, attention_masks, num_return_sequences):
        beam_outputs = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_masks,
            do_sample=True,
            max_length=self.max_len,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=num_return_sequences)

        generated_sentences = []
        for output in beam_outputs:
            generated_sentences.append(self.tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True))
        
        return generated_sentences

    def _get_least_similar_paraphrased_sentence(self, sentence, paraphrased_sentences):
        encoded_sentence = self.embedding_model.encode(sentence)
        
        min_similarity = 1
        min_paraphrased_sentence = None
        for paraphrased_sentence in paraphrased_sentences:
            if paraphrased_sentence == sentence:
                continue

            encoded_paraphrased_sentence = self.embedding_model.encode(paraphrased_sentence)
            similarity = self._get_cosine_similarities(encoded_sentence, encoded_paraphrased_sentence)
            
            if similarity < min_similarity:
                min_similarity = similarity
                min_paraphrased_sentence = paraphrased_sentence
        
        return min_paraphrased_sentence

    def _get_cosine_similarities(self, claim_embedding, sentence_embedding):
        return 1 - spatial.distance.cosine(claim_embedding, sentence_embedding)


    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
