# NLP Homework 07
#### Yunhao Li, NetID: yl6220

## Dev score with and without the embeddings
+ Without embeddings: 82.93
+ With embeddings:
+ With additional methods and embeddings: 

## Base
   + Implemented the binarization with a fixed threshold.
## The additional methods:
   + Implemented the binarization described in _Revisiting Embedding Features for Simple Semi-supervised Learning_.
   + Implemented the cluster method.<br>
      I used the K-means method provided by `sklearn`. And running the kmeans on the word embedding vectors of 
      glove.6B.50d.txt. After training I use the trained model to predict the class for each word, and add its class as 
      a feature. I tried several `N_cluster` params and finally choose `N_cluster = 20`.<br>
      N_cluster = 10: F1 = <br>
      N_cluster = 15: F1 = <br>
      N_cluster = 20: F1 = 84.94 <br>
      N_cluster = 30: F1 = 84.77 <br>
## Required library
   + sklearn
   + numpy
