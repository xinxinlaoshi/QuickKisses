# QuickKisses
## Table of Contents
* [Introduction](#introduction)
* [Setup](#setup)
* [Corpus file](#corpus-file)
* [Experiment 1](#experiment-1)
* [Experiment 2](#experiment-2)

## Introduction
This project aims to carry out the two experiments in the paper *On Quick Kisses and How to Make Them Count:
A Study on Event Construal in Light Verb Constructions with BERT*.  


## Setup
To successfully replicate the experiments, install the following libraries from Pypi using:  
`pip install minicons`  
`pip install stanza`

Decompress the target sentences zip folder under the main directory and decompress the feature words zip folder under the directory of Experiment_1

## Corpus file
This project uses the British National Corpus file in XML format (BNC Consortium, 2007, *British National Corpus*, *XML edition*, Oxford Text Archive, http://hdl.handle.net/20.500.12024/2554.)

## Experiment 1
#### This experiment uses semantic projection to reproduce the results of Experiment 1-2 in Wittenberg & Levy (2017)'s study
`feature_words_extraction.py`: run this file to extract the sentences where the feature words occur in natural contexts from the British National Corpus

`feature_vector_aggregation.py`: run this file to aggregate the vectors of the feature words into a 1-dimensional subspace
### Code example
 initialize the Transformer model and import the feature words into a dictionary   
 
```
tf = feature_vector('bert-base-uncased')
tf.dict = tf.get_feature_words('.\\feature_words') 
```

if run the projection in setting (i)  
```
# select the words used to represent the concept 'long'
long_list = ['long', 'long-term', 'ages', 'years', 'centuries', 'lengthy', 'decades']

# select the words used to represent the concept 'short'
short_list = ['brief', 'immediate', 'minute',' moment', 'second', 'short', 'short-term']

c_vector = tf.dimension_vector(1000, long = long_list, short = short_list)

print(c_vector)
'''
tensor([ 3.1756e+00,  2.0101e+00,  3.0391e+00, ...,  -1.7005e+00, -5.6126e-01,  1.4112e-01])
'''
```
`semantic_projection`: run this file to project the target events onto the aggregated feature vector

### Code example
import the target events
```
Root_ditransitive = '..\\target_sentences\ditransitive'
Root_transitive = '..\\target_sentences\\transitive'

ditransitive_sen = read_sentences(Root_ditransitive)
transitive_sen = read_sentences(Root_transitive)
```

import the aggregated feature vector
```
c_vector = torch.load('.\\feature_vectors\proj_i.pt') 
# c_vector = torch.load('.\\feature_vectors\proj_ii.pt')
```

initialize the model
```
tv = transformer_vector('bert-base-uncased')
```

implement the projection for target events in ditransitive constructions
```
projections = {}
for word, sentences in ditransitive_sen.items():
  import random
  random.shuffle(sentences)
  projection_score = tv.semantic_projection(word, sentences, 40, 'ditransitive', c_vector)
  projections[word] = projection_score
```

implement the projection for target events in transitive constructions
```
projections2 = {}
for word, sentences in transitive_sen.items():
  import random
  random.shuffle(sentences)
  projection_score = tv.semantic_projection(word, sentences, 40, 'transitive', c_vector)
  projections2[word] = projection_score
  ```

Transform the data into the data frame format and import it into a table  (ditransitive construction as an example)
```
lst = [projections]
df = pd.DataFrame(lst).T
df = df.reset_index()
df.columns = ['event','projection']
df['event'] = df['event'].apply(vn)
df['event category'] = df['event'].apply(event_category)

'''
>>> vn('present') # vn converts the verb to its noun form
presentation
>>> event_category('support') # event_category classifies the event into its corresponding event category
durative mass
'''
```

| event | projection | event category |
|-------|------------|----------------|
| kiss | -0.75331 | punctive count |
| wink | -0.63574 | punctive count |
| ...... | ...... | ...... | 
| address | -0.42327 | durative count |
| ...... | ...... | ...... |
| thanks | 0.464643 | durative mass |


## Experiment 2
#### This experiment measures the semantic similarity of target events in different contexts to reproduce the results of Experiment 4 in Wittenberg & Levy (2017)'s study
`semantic_similarity.py`: sample pairs of target events in natural contexts in transitive constructions / ditransitive constructions / both transitive and ditransitive constructions

### Code example
measure the semantic similarity of context pairs sampled both in ditransitive contexts
```
model_name = 'bert-base-uncased'
similarity_ditransitive = model_running('n',model_name, ditransitive_sen)
```
measure the semantic similarity of context pairs sampled both in transitive contexts
```
similarity_transitive = model_running('v',model_name, transitive_sen)
```
measure the semantic similarity of context pairs composed of one occurrence in the ditransitive context and one in the transitive context
```
similarity_both = model_running('both',model_name, ditransitive_sen, transitive_sen)
```
Transform the data into the data frame format and import it into a table 
```
df1 = pd.DataFrame(similarity_ditransitive).T
df1 = df1.reset_index()
df1.columns = ['event','spearman','cosine']
df1['event'] = df1['event'].apply(vn)
df1['event category'] = df1['event'].apply(event_category)
```
| event | event category | spearman | cosine |
|-------|----------------|----------|--------|
| address | durative count | 0.496008 | 0.572848 |
| ...... | ...... | ...... | ...... |
