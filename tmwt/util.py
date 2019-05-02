from gensim import utils, matutils
from gensim.models.phrases import Phraser
import os
import re
from unidecode import unidecode

  
BASE_DIRECTORY = './' 
MALLET_PATH = '/opt/code/rtt/git/mallet-2.0.8/bin/mallet'

DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'data')
MODEL_DIRECTORY = os.path.join(BASE_DIRECTORY, 'model')
CATEGORY_INFO_PATH = os.path.join(MODEL_DIRECTORY, 'cat-subcat.tsv')
DICTIONARY_PATH = os.path.join(MODEL_DIRECTORY, 'dictionary.pkl')
SIMILARITY_INDEX_PATH = os.path.join(MODEL_DIRECTORY, 'similarity.index')
MALLET_MODEL_PATH = os.path.join(MODEL_DIRECTORY, 'mallet.model')
GENSIM_MODEL_PATH = os.path.join(MODEL_DIRECTORY, 'gensim.model')

MIN_DOCUMENT_LENGTH = 200
MAX_DOCUMENT_LENGTH = 10000

def clean(text, max_len=MAX_DOCUMENT_LENGTH):
    text = text.strip()
    if max_len is not None: text = text[:max_len]
    text = unidecode(text)
    text = re.sub("\s+", " ", text)
    return text

def load_records(path=None, min_len=MIN_DOCUMENT_LENGTH, max_len=MAX_DOCUMENT_LENGTH):
    if path is None: path = os.path.join(DATA_DIRECTORY, "corpus.tsv")
    with open(path, mode='r', encoding='UTF-8') as ifp: lines = ifp.readlines()
    records = []
    bad_record_cnt = 0
    for line in lines:
        line = line.strip()
        if line:
            try:
                record_id, text = line.split('\t')
                text = clean(text, max_len)
                if min_len is None or len(text) > min_len: records.append((record_id, text))
                else: bad_record_cnt += 1
            except: bad_record_cnt += 1
    print("Loaded %d records with %d errors." % (len(records), bad_record_cnt))
    return records

def load_phrasers(directory=MODEL_DIRECTORY):
    path = os.path.join(directory, "bigram-phraser.pkl")
    bigram_phraser = Phraser.load(path)

    path = os.path.join(directory, "trigram-phraser.pkl")
    trigram_phraser = Phraser.load(path)
    
    return bigram_phraser, trigram_phraser

def is_token(dictionary, i):
    word = dictionary[i]
    if '-' in word or '_' in word: return False
    return True

def is_not_token(dictionary, i):
    return not is_token(dictionary, i)

def best_items(topic, dictionary, f, n):
    data = [(i, score) for i, score in enumerate(topic) if score > 0 and f(dictionary, i)]
    indices, scores = zip(*data)
    return [indices[i] for i in matutils.argsort(scores, n, reverse=True)]

def represent(dictionary, model, id, n=10, m=6, num_words=None, indent="", use_phrasers=False):
    topic = model.word_topics[id]
    topic = topic / topic.sum()
    top_tokens = [dictionary[i] for i in best_items(topic, dictionary, is_token, n)]
    if use_phrasers:
        top_phrases = [dictionary[i] for i in best_items(topic, dictionary, is_not_token, m)]
        return "%s%s\n%s%s" % (indent, ", ".join(top_tokens), indent, ", ".join(top_phrases))
    return "%s%s" % (indent, ", ".join(top_tokens))

def show_topic_model(model, dictionary, use_phrasers=False):
    for i in range(model.num_topics): 
        print("Topic %d:" % i)    
        print(represent(dictionary, model, i, indent="  ", use_phrasers=use_phrasers))

def represent_lda(dictionary, topics, id, n=10, m=6, num_words=None, indent="", use_phrasers=False):
    topic = topics[id]
    topic = topic / topic.sum()
    top_tokens = [dictionary[i] for i in best_items(topic, dictionary, is_token, n)]
    if use_phrasers:
        top_phrases = [dictionary[i] for i in best_items(topic, dictionary, is_not_token, m)]
        return "%s%s\n%s%s" % (indent, ", ".join(top_tokens), indent, ", ".join(top_phrases))
    return "%s%s" % (indent, ", ".join(top_tokens))

def show_topic_lda(lda, dictionary, use_phrasers=True):
    topics = lda.state.get_lambda()
    for i in range(lda.num_topics): 
        print("Topic %d:" % i)    
        print(represent_lda(dictionary, topics, i, indent="  ", use_phrasers=use_phrasers))
        
def show_document_topics(vector, n=10):
    vector = sorted(vector, key=lambda item:-item[-1])
    print(" + ".join(["%0.3f * %02d" % (w, i) for i, w in vector[:n]]))


