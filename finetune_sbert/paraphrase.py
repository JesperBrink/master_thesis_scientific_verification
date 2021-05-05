def paraphrase_claim_and_sentences(claim, sentences):
    claim_and_sentence_pairs = []

    paraphrased_claim = paraphrase(claim)
    paraphrased_sentences = [paraphrase(sentence) for sentence in sentences]
    
    # (claim, sentence)
    for sentence in sentences:
        claim_and_sentence_pairs.append((claim, sentence))

    # (paraphrased claim, sentence)
    for sentence in sentences:
        claim_and_sentence_pairs.append((paraphrased_claim, sentence))

    # (claim, paraphrased sentence)
    for paraphrased_sentence in paraphrased_sentences:
        claim_and_sentence_pairs.append((claim, paraphrased_sentence))

    # (paraphrased claim, paraphrased sentence)
    for paraphrased_sentence in paraphrased_sentences:
        claim_and_sentence_pairs.append((paraphrased_claim, paraphrased_sentence))

    return claim_and_sentence_pairs


def paraphrase(text):
    return text
