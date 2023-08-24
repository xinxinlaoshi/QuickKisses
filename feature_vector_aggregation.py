import numpy as np
from minicons import cwe
import os, re
import torch
from transformers import BertTokenizer
class feature_vector(object):

    def __init__(self, model_name):
        self.model = cwe.CWE(model_name, device='cpu')
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def get_feature_words(self,file_dir):
        self.dict = {}
        for root, dirs, files in os.walk(file_dir):
          for file in files:
            data_dir = os.path.join(root, file)
            try:

                key = re.findall(r'\\([a-z-]+).npz', data_dir)[0]
                data = np.load(data_dir)
                lst = data['arr_0']
                lst = lst.tolist()
                self.dict[key] = lst
            except Exception as error:
              print(error)
              print(data_dir)
              print('error in extracting feature words')
        return self.dict

    def get_vector(self, word, sentence):
        return self.model.extract_representation([sentence, word])

    def get_sentence(self, word):
        import random
        sentences = self.dict[word]
        random.shuffle(sentences)
        return sentences

    def get_centroid(self, word, n):
        """
        calculate the centroid vector of multiple word vectors
        :param words: word list
        :return: the centroid vector of the target word aggregated from n sentences
        """
        container = torch.zeros(768)  # 768 is the dimesionality of the hidden layers in base version
        sentences = self.get_sentence(word)  # n = 1000
        count = 0
        for sentence in sentences:
          if count > n:
            break
          sentence = sentence.lower()
          tokenized_sentence = self.tokenizer.tokenize(sentence)
          if len(tokenized_sentence) > 510:
            continue
          try:
            vector = self.get_vector(word, sentence)
            assert torch.isnan(vector).any() == False
            container = container + vector[0]
            count += 1
          except Exception as error:
             print(word)
             print(sentence)
             print(error)


        return torch.divide(container, n)

    def dimension_vector(self, n, **features):

        vector_long = torch.zeros(768)
        vector_short = torch.zeros(768)

        for f in features['long']:
            print('generating the vector for:',f)
            vector = self.get_centroid(f, n)
            assert torch.isnan(vector).any() == False
            vector_long = vector_long + vector

        no_1 = int(len(features['long'])) # averaging the vectors representing the feature "long"
        self.vector_long = torch.divide(vector_long, no_1)
        assert torch.isnan(self.vector_long).any() == False

        for f in features['short']:
            vector = self.get_centroid(f, n)
            print('generating the vector for:', f)
            assert torch.isnan(vector).any() == False
            vector_short = vector_short + vector

        no_2 = int(len(features['short'])) # averaging the vectors representing the feature "short"
        self.vector_short = torch.divide(vector_short, no_2)
        assert torch.isnan(self.vector_short).any() == False

        c_vector = torch.subtract(vector_long, vector_short)
        assert torch.isnan(c_vector).any() == False
        return c_vector


tf = feature_vector('bert-base-uncased')
tf.dict = tf.get_feature_words('.\\feature_words')

# projection condition (i)
long_list = ['long','long-term','ages','years','centuries','lengthy','decades']
short_list = ['brief','immediate','minute','moment','second','short','short-term']

# projection condition (ii)
# long_list = ['long','long-term','years']
# short_list = ['immediate','minute','moment']

c_vector = tf.dimension_vector(1000, long = long_list, short = short_list)
torch.save(c_vector, '.\\feature_vectors\proj_i.pt')
# torch.save(c_vector, '.\\features_vectors\proj_ii.pt')










