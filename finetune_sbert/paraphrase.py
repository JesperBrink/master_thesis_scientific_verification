import json
import logging
from pathlib import Path
import sys

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class Paraphraser():
    def __init__(self):
        self._set_seed(42)
        self.tokenizer = T5Tokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws')
        self.model = T5ForConditionalGeneration.from_pretrained('Vamsi/T5_Paraphrase_Paws')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.max_len = 256

    def paraphrase_claim_and_sentences(self, claim, sentences):
        claim_and_sentence_pairs = []

        paraphrased_claim = self._paraphrase(claim)
        paraphrased_sentences = [self._paraphrase(sentence) for sentence in sentences]

        if paraphrased_claim is None:
            return []
        
        for sentence in sentences:
            if sentence is None:
                break
            claim_and_sentence_pairs.append((claim, sentence))
            claim_and_sentence_pairs.append((paraphrased_claim, sentence))

        for paraphrased_sentence in paraphrased_sentences:
            if paraphrased_sentence is None:
                break
            claim_and_sentence_pairs.append((claim, paraphrased_sentence))
            claim_and_sentence_pairs.append((paraphrased_claim, paraphrased_sentence))

        return claim_and_sentence_pairs

    def _paraphrase(self, sentence):
        text = "paraphrase: " + sentence
        
        encoding = self.tokenizer.encode_plus(text, padding=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding[
            "attention_mask"].to(self.device)

        paraphrased_sentence = None
        counter = 0
        while paraphrased_sentence is None or paraphrased_sentence.lower() == sentence.lower():
            paraphrased_sentence = self._generate_paraphrased_sentence(sentence, input_ids, attention_masks)            
            counter += 1
            if counter == 5:
                return None

        return paraphrased_sentence

    def _generate_paraphrased_sentence(self, sentence, input_ids, attention_masks):
        beam_outputs = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_masks,
            do_sample=True,
            max_length=self.max_len,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1)

        beam_output = beam_outputs[0]
        return self.tokenizer.decode(
            beam_output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
