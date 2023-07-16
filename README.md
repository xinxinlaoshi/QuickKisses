# QuickKisses
## Table of Contents
* [Introduction] (#introduction)
* [Setup] (#setup)
* [Corpus file] (#corpus-file)
* [Experiment 1: semantic projection] (#experiment-1:-semantic-projection)
* [Experiment 2: semantic similarity] (#experiment-2:-semantic-similarity)

## Introduction
This project aims to carry out the two experiments in the paper *On Quick Kisses and How to Make Them Count:
A Study on Event Construal in Light Verb Constructions with BERT*.  


## Setup
To successfully replicate the experiments, install the following libraries from Pypi using:
`pip install minicons`
`pip install stanza`

## Corpus file
This project uses the British National Corpus file in XML format (BNC Consortium, 2007, *British National Corpus*, *XML edition*, Oxford Text Archive, http://hdl.handle.net/20.500.12024/2554.)

## Experiment 1: semantic projection
`feature_words_extraction.py`: run this file to extract the sentences where the feature words occur in natural contexts from the British National Corpus

`feature_vector_aggregation.py`: run this file to aggregate the vectors of the feature words into a 1-dimensional subspace
### Code example
` tf = feature_vector('bert-base-uncased') # initialize the Transformer model
tf.dict = tf.get_feature_words('.\\feature_words') # import the feature words into a dictionary`
` # if run the projection in setting (i)
long_list = ['long', 'long-term', 'ages', 'years', 'centuries', 'lengthy', 'decades'] # select the words used to represent the concept 'long' 
short_list = ['brief', 'immediate', 'minute',' moment', 'second', 'short', 'short-term'] #select the words used to represent the concept 'short'
c_vector = tf.dimension_vector(1000, long = long_list, short = short_list)
print(c_vector)
'''
tensor([ 3.1756e+00,  2.0101e+00,  3.0391e+00, ...,  -1.7005e+00, -5.6126e-01,  1.4112e-01])
'''
`
`semantic_projection1`: run this file to project the target events onto the aggregated feature vector
### Code example
import the target events and the feature vector
`Root_ditransitive = '..\\target_sentences\ditransitive'
Root_transitive = '..\\target_sentences\\transitive'
ditransitive_sen = read_sentences(Root_ditransitive)
transitive_sen = read_sentences(Root_transitive) 
c_vector = torch.load('.\\feature_vectors\proj_i.pt')
`
initialize the model
`tv = transformer_vector('bert-base-uncased')`

implement the projection for target events in ditransitive constructions
`projections = {}
for word, sentences in ditransitive_sen.items():
  import random
  random.shuffle(sentences)
  projection_score = tv.semantic_projection(word, sentences, 40, 'ditransitive', c_vector)
  projections[word] = projection_score
`
implement the projection for target events in transitive constructions
`projections2 = {}
for word, sentences in transitive_sen.items():
  import random
  random.shuffle(sentences)
  projection_score = tv.semantic_projection(word, sentences, 40, 'transitive', c_vector)
  projections2[word] = projection_score
  `
Import the data in the table (ditransitive construction as an example)
`
| event | projection | event category |
|-------|------------|----------------|
| kiss | -0.75331 | punctive count |
| wink | -0.63574 | punctive count |
| ...... | ...... | ...... | 
| address | -0.42327 | durative count |
| ...... | ...... | ...... |
| thanks | 0.464643 | durative mass |
`
## Experiment 2: semantic similarity
`semantic_similarity.py`: sample pairs of target events in natural contexts in transitive constructions / ditransitive constructions / both transitive and ditransitive constructions





