from minicons import cwe
import torch
from transformers import BertTokenizer
import spacy
import pandas as pd

class transformer_vector(object):
    def __init__(self, model_name):
        self.model = cwe.CWE(model_name, device='cpu')
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def get_vector(self, word, sentence):
        return self.model.extract_representation([sentence, word])

    def semantic_projection(self, word, sentences, N, type,c_vector):
        '''

        :param word: the target event, i.e. kiss
        :param sentences: list of sentences containing the target event
        :param N: the number of sentences used for projection
        :param type: the construction (transitive or ditransitive)
        :param c_vector: the feature vector
        :return: the average projection score
        '''
        from nltk import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        nlp = spacy.load("en_core_web_sm")

        concept_norm = torch.norm(c_vector).item()
        assert concept_norm != 0

        score = 0
        n = 0
        for sentence in sentences:
            if n == N:
                break
            sentence = sentence.lower()
            try:
                doc = nlp(sentence)
                words = [w.text for w in doc]
                if type == 'ditransitive':
                    pos = 'n'
                elif type == 'transitive':
                    pos = 'v'
                event = [t for t in words if wnl.lemmatize(t, pos) == word]
                event = event[0]
                tokenized_sentence = self.tokenizer.tokenize(sentence)
                assert len(tokenized_sentence) < 510
            except Exception as error:
                print('error in finding the event:', error)
                print(words)
                continue

            target_vector = self.get_vector(event, sentence)[0]
            projection_score = torch.dot(target_vector, c_vector).item() / concept_norm
            score = score + round(projection_score, 4)
            n += 1

        return score / N


def read_sentences(file_root):
    import os, re
    import numpy as np
    sentence_dict = {}
    for root, dirs, files in os.walk(file_root):
        for file in files:
            data_dir = os.path.join(root, file)
            key = re.findall(r'\\([a-z-]+).npz', data_dir)[0]
            data = np.load(data_dir)
            lst = data['arr_0'].tolist()
            sentence_dict[key] = []
            for line in lst:
                sentence_dict[key].append(line)
    return sentence_dict

Root_ditransitive = '..\\target_sentences\ditransitive'
Root_transitive = '..\\target_sentences\\transitive'
ditransitive_sen = read_sentences(Root_ditransitive)
transitive_sen = read_sentences(Root_transitive)


c_vector = torch.load('.\\feature_vectors\proj_i.pt')

tv = transformer_vector('bert-base-uncased')

projections = {}
for word, sentences in ditransitive_sen.items():
  import random
  random.shuffle(sentences)
  projection_score = tv.semantic_projection(word, sentences, 40, 'ditransitive', c_vector)
  projections[word] = projection_score

projections2 = {}
for word, sentences in transitive_sen.items():
  import random
  random.shuffle(sentences)
  projection_score = tv.semantic_projection(word, sentences, 40, 'transitive', c_vector)
  projections2[word] = projection_score

def event_category(string):
  DM = ['support','thanks','recognition','assurance','advice','encouragement']
  DC = ['talk','address','lecture','presentation','present','speech','speak','check']
  PC = ['kiss','hug','kick','shake','cuddle','wink']
  if string in DM:
    return "durative mass"
  elif string in DC:
    return "durative count"
  elif string in PC:
    return "punctive count"

def vn(string):
  verb_noun = {'kiss':'kiss','hug':'hug','kick':'kick','shake':'shake','cuddle':'cuddle','wink':'wink','talk':'talk','address':'address','lecture':'lecture','present':'presentation','speak':'speech','check':'check','support':'support','thank':'thanks','recognize':'recognition','assure':'assurance','advise':'advice','encourage':'encouragement'}
  if string in verb_noun.keys():
    string = verb_noun[string]
  return string

ditransitive_file = "ditransitive.csv"
transitive_file = "transitive.csv"

lst = [projections]
df = pd.DataFrame(lst).T
df = df.reset_index()
df.columns = ['event','projection']
df['event'] = df['event'].apply(vn)
df['event category'] = df['event'].apply(event_category)
df.to_csv(ditransitive_file)

lst2 = [projections2]
df2 = pd.DataFrame(lst2).T
df2 = df2.reset_index()
df2.columns = ['event','projection']
df2['event'] = df2['event'].apply(vn)
df2['event category'] = df2['event'].apply(event_category)
df2.to_csv(transitive_file)

















