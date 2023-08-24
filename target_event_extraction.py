import os, stanza
import xml.etree.ElementTree as ET

en_nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True)

def if_ditransitive(string):
    noun = ['kiss', 'hug', 'kick', 'shake', 'cuddle', 'wink', 'talk', 'address', 'lecture', 'presentation', 'speech',
            'check', 'support', 'thanks', 'recognition', 'assurance', 'advice', 'encouragement']
    storage = {}
    mass = ['assurance', 'support', 'encouragement', 'thanks', 'recognition', 'advice']
    for n in noun:
        storage[n] = []
    doc = en_nlp(string)

    for sen in doc.sentences:
        sentence = sen.to_dict()
        try:
            head_id = [word["id"] for word in sentence if word['lemma'] == 'give']

            for word in sentence:
                if word['lemma'] == 'thanks' and word['head'] in head_id and word["deprel"] == "obj":
                    storage[word['lemma']].append(sen.text)
                elif word['lemma'] in noun and word['head'] in head_id and word["deprel"] == "obj":
                    if word['lemma'] in mass and word['xpos'] == 'NN':
                        storage[word['lemma']].append(sen.text)
                    elif word['lemma'] not in mass:
                        if word['xpos'] == 'NNS':
                            storage[word['lemma']].append(sen.text)
                        else:
                            det = [item['lemma'] for item in sentence if
                                   item['head'] == word['id'] and item['deprel'] == 'det']
                            if 'a' in det or 'an' in det:
                                storage[word['lemma']].append(sen.text)

        except Exception as error:
            print(error)
            print(word['lemma'])
            print(sen)

    return storage

corpus_root = '.\BNC' #the root of the corpus file
noun = ['kiss','hug','kick','shake','cuddle','wink','talk','address','lecture','presentation','speech','check','support','thanks','recognition','assurance','advice','encouragement']
verb = ['kiss','hug','kick','shake','cuddle','wink','talk','address','lecture','present','speak','check','support','thank','recognize','assure','advise','encourage']

transitive_sen = {}
for v in verb:
    transitive_sen[v] = []
string = ''

Filelist = []
for root, dirs, files in os.walk(corpus_root):
    for file in files:
        Filelist.append(os.path.join(root,file))

for f in Filelist:
    tree = ET.parse(f)
    root = tree.getroot()

    for sentence in root.iter('s'):
        give = False
        hw = False

        lst = [words.attrib for words in sentence.iter(None) if 'hw' in words.attrib.keys()]

        for item in lst:
            if item['hw'] == 'give':
                give = True
            if item['hw'] in noun:
                hw = True
            if item['hw'] in verb and item['pos'] == 'VERB':
                sen = ''.join([w for w in sentence.itertext()])
                transitive_sen[item['hw']].append(sen)

        if give == True and hw == True: # this step is to extract sentences where 'give' co-occurs with the target nouns
            sen = ''.join([w for w in sentence.itertext()])
            string = string + sen + '\n\n'


ditransitive_sen = if_ditransitive(string)

import numpy as np
for key, value in ditransitive_sen.items():
  dir = ".\\target_sentences\ditransitive\{}".format(key)
  np.savez(dir,value)

for key, value in transitive_sen.items():
  dir = ".\\target_sentences\\transitive\{}".format(key)
  np.savez(dir,value)









