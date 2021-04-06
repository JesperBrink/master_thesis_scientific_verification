from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

def lemmatization(doc):
    lemmatizer = WordNetLemmatizer()

    for sentence_idx, sentence in enumerate(doc["abstract"]):
        sent = []
        for w in word_tokenize(sentence):
            # NOTE: To get pos tag we need to tokenize, which may not be desirable, as we then do more than one thing
            for word, pos in pos_tag(w): 
                # Simply lemmatizing everything may give wrong results according to SO, as the lemmatizer assumes noun if no POS tag is given
                if get_wordnet_pos(pos) is not None:
                    word = lemmatizer.lemmatize(word, get_wordnet_pos(pos))
                
                sent.append(word)
        doc["abstract"][sentence_idx] = " ".join(sent)

    return doc


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
