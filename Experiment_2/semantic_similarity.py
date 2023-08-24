# root of the files storing the target sentences extracted from BNC
Root_ditransitive = '..\\target_sentences\ditransitive'
Root_transitive = '..\\target_sentences\\transitive'

# root to store the files containing the computation results of the semantic similarity
ditranSimi = '.\similarity_results\ditransitive_pair.csv'
tranSimi = '.\similarity_results\\transitive_pair.csv'
bothSimi = '.\similarity_results\\both.csv'


# if the input has one list, then the output is a list consisting of random sentence pairs
# if the input has two lists, then the output is a list consisting of pairs of the sentences extracted from each input list respectively
def select_random_pair(*lst):
    import random
    if len(lst) == 1:  # if the input is only one list
        random.shuffle(lst[0])
        result = [lst[0][i:i + 2] for i in range(0, len(lst[0]), 2)]
        return result
        # result = [[sen1, sen2], [sen3, sen4], ...]

    else:  # if the input contains two lists
        random.shuffle(lst[0])
        random.shuffle(lst[1])
        try:
            result = [[s1, s2] for s1, s2 in zip(lst[0], lst[1])]
            return result
        except:
            pass



def semantic_similarity(word, lst, model, n, pos):
    '''
    :param word: target event
    :param lst: a list consisting of sentence pairs
    :param model: bert-base-uncased, bert-large-uncased, distilbert-base-uncased;
    :param n: the number of pairs used to compute the average semantic similarity
    :param pos: v(when the sentence contains transitive construction), n (ditransitive), both (both transitive and ditransitive)
    :return: the average spearman correlation and cosine similarity value
    '''

    import torch, spacy
    import numpy as np
    from scipy import stats, spatial
    from nltk import WordNetLemmatizer

    nlp = spacy.load("en_core_web_sm")
    wnl = WordNetLemmatizer()
    array = np.array([0, 0])  # the array will be used to store the spearman correlation and cosine similarity
    noun_verb = {'kiss': 'kiss', 'hug': 'hug', 'kick': 'kick', 'shake': 'shake', 'cuddle': 'cuddle', 'wink': 'wink',
                 'talk': 'talk', 'address': 'address', 'lecture': 'lecture', 'presentation': 'present',
                 'speech': 'speak', 'check': 'check', 'support': 'support', 'thanks': 'thank',
                 'recognition': 'recognize', 'assurance': 'assure', 'advice': 'advise', 'encouragement': 'encourage'}

    count = 0
    for pair in lst:
        if count == n:
            break

        if len(pair) == 1:
            print(count)
            break

        s1, s2 = pair[0].lower(), pair[1].lower()
        doc1, doc2 = nlp(s1), nlp(s2)
        words1 = [w.text for w in doc1]
        words2 = [w.text for w in doc2]
        try:
            if pos == 'both':
                target1 = [t for t in words1 if wnl.lemmatize(t, 'n') == word][0]
                target2 = [t for t in words2 if wnl.lemmatize(t, 'v') == noun_verb[word]][0]

            else:
                target1 = [t for t in words1 if wnl.lemmatize(t, pos) == word][0]
                target2 = [t for t in words2 if wnl.lemmatize(t, pos) == word][0]
        except Exception as error:
            print('sampling error:', error)
            break
            # continue

        try:
            instance = [[s1, target1], [s2, target2]]
            word_rep = model.extract_representation(instance)
            assert torch.isnan(word_rep).any() == False  # to prevent that tensor contains nan

            score_spearman = stats.spearmanr(word_rep[0], word_rep[1])[0]
            score_cosine = 1 - spatial.distance.cosine(word_rep[0], word_rep[1])

            array1 = np.array([score_spearman, score_cosine])
            array = np.vstack([array, array1])

            count += 1

        except Exception as error:
            print(error)
            print(instance)

    score = np.delete(array, [0, 0], axis=0)
    spearman = np.mean(score, axis=0)[0]
    cosine = np.mean(score, axis=0)[1]
    return [spearman, cosine]


# read the sentences and store them in a dictionary
def read_sentences(file_root):
    import os, re
    import numpy as np
    sentence_dict = {}
    for root, dirs, files in os.walk(file_root):
        for file in files:
            data_dir = os.path.join(root, file)

            key = re.findall(r'\\([a-z]*).npz', data_dir)[0]
            data = np.load(data_dir)
            lst = data['arr_0'].tolist()
            sentence_dict[key] = []
            for line in lst:
                sentence_dict[key].append(line)
    return sentence_dict


ditransitive_sen = read_sentences(Root_ditransitive)
transitive_sen = read_sentences(Root_transitive)


from minicons import cwe  # import the pre-installed minicons library

def model_running(i,model, *lst):#i: noun or verb
    import numpy as np
    similarity = {}
    noun_verb = {'kiss': 'kiss', 'hug': 'hug', 'kick': 'kick', 'shake': 'shake', 'cuddle': 'cuddle', 'wink': 'wink',
                 'talk': 'talk',
                 'address': 'address', 'lecture': 'lecture', 'presentation': 'present', 'speech': 'speak',
                 'check': 'check', 'support': 'support',
                 'thanks': 'thank', 'recognition': 'recognize', 'assurance': 'assure', 'advice': 'advise',
                 'encouragement': 'encourage'}

    for k in lst[0].keys():
        arr = np.array([0, 0])
        count = 0
        while count < 10:
          try:
              if len(lst) == 1:
                pairs = select_random_pair(lst[0][k])
                score = semantic_similarity(k, pairs, model, 10, i)
              else:
                v = noun_verb[k]
                pairs = select_random_pair(lst[0][k],lst[1][v])
                score = semantic_similarity(k, pairs, model, 10, i)

              arr1 = np.array(score)
              arr = np.vstack([arr, arr1])
              count += 1

          except Exception as error:
              print('running error:', error)
              continue

          results = np.delete(arr, [0, 0], axis=0)
          spearman = np.mean(results, axis=0)[0]
          cosine = np.mean(results, axis=0)[1]
          pair = [spearman, cosine]
          similarity[k] = pair

    return similarity


import pandas as pd

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

model_name = 'bert-base-uncased'

similarity_ditransitive = model_running('n',model_name, ditransitive_sen)
df1 = pd.DataFrame(similarity_ditransitive).T
df1 = df1.reset_index()
df1.columns = ['event','spearman','cosine']
df1['event'] = df1['event'].apply(vn)
df1['event category'] = df1['event'].apply(event_category)
df1.to_csv(ditranSimi)

similarity_transitive = model_running('v',model_name, transitive_sen)
df2 = pd.DataFrame(similarity_transitive).T
df2 = df2.reset_index()
df2.columns = ['event','spearman','cosine']
df2['event'] = df2['event'].apply(vn)
df2['event category'] = df2['event'].apply(event_category)
df2.to_csv(tranSimi)

similarity_both = model_running('both',model_name, ditransitive_sen, transitive_sen)
df3 = pd.DataFrame(similarity_both).T
df3 = df3.reset_index()
df3.columns = ['event','spearman','cosine']
df3['event'] = df3['event'].apply(vn)
df3['event category'] = df3['event'].apply(event_category)
df3.to_csv(bothSimi)
