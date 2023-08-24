import os, re
import xml.etree.ElementTree as ET
import numpy as np

corpus_root = '..\BNC'
Filelist = []
for root, dirs, files in os.walk(corpus_root):
   for file in files:
        Filelist.append(os.path.join(root,file))

adj = ['long','long-term','lengthy','brief','immediate','short','short-term']
plural = ['age','year','decade','century']
singular = ['minute','moment','second']
features = {}
for f in Filelist:
    tree = ET.parse(f)
    root = tree.getroot()

    for sentence in root.iter('s'):
        lst = [words.attrib for words in sentence.iter(None) if 'hw' in words.attrib.keys()]

        string = ''.join([w for w in sentence.itertext()])

        for items in lst:
            if items['hw'] in adj:
                if items['c5'] == 'AJ0':
                    features[items['hw']].append(string)
            elif items['hw'] in plural:
                if items['c5'] == 'NN2':
                    features[items['hw']].append(string)
            elif items['hw'] in singular:
                if items['c5'] == 'NN1':
                    features[items['hw']].append(string)
            elif items['hw'] == 'long' or items['hw'] == 'short':
                if re.findall('(:?[Ll]ong|[Ss]hort) (?:time|period)', string):
                    features[items['hw']].append(string)

for key, value in features.items():
  dir = ".\\feature_words\{}".format(key)
  np.savez(dir,value)




















