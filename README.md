# NLP Homework 07
#### Yunhao Li, NetID: yl6220

## Best dev score with and without the embeddings
+ Without embeddings: F1 = 82.93
+ With embeddings(fixed threshold): F1 = 
+ With additional methods and embeddings: 
  + With binarization:
    
  + With cluster:<br>
    N_cluster = 20<br>
    F1 = 84.94
## Base
   + Implemented the binarization with a fixed threshold.<br>
   I use a single threshold to binarilize all 50 dimensions.<br>
   And the threshold I find best is:<br>
   Threshold = 0.1
   F1 = 
   
## The additional methods:
   + Implemented the binarization described in _Revisiting Embedding Features for Simple Semi-supervised Learning_.
      For each dimension, I calculate the mean of all the positive values and the mean of the negative values. Then I 
      use these two means `(trshd_pos[0..49], trshd_neg[0..49])` as the threshold of the value. If the value `vec[i]` 
      that is greater than `trshd_pos[i]`, then use `+1` as the binarized result; or if it is less than `trshd_neg[i]`, 
      then use `-1` as the result; or the result is `0`. And finally I put the 50 dimensions of the word as 50 features 
      and add them into the feature file to train the model or get tagged.<br>
   <br>
   + Implemented the cluster method.<br>
      I used the K-means method provided by `sklearn`. And running the kmeans on the word embedding vectors of 
      glove.6B.50d.txt. After training I use the trained model to predict the class for each word, and add its class as 
      a feature. I tried several `N_cluster` params and finally choose `N_cluster = 20`.<br>
      
## Required library
   + sklearn
   + numpy
