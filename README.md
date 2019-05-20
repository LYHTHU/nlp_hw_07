# NLP Homework 07
#### Yunhao Li, NetID: 

## Best dev score with and without the embeddings
+ Without embeddings: F1 = 82.93<br>
    <br>
+ With embeddings(fixed threshold): F1 = 80.29<br>
    <br>
+ With additional methods and embeddings: 
  + With binarization:<br>
    F1 = 80.01<br>
    <br>
  + With cluster:<br>
    N_cluster = 20<br>
    F1 = 84.94<br>
    **Note:** the result of cluster is not fixed every time. Because the K-means has some stochastic process.
## Base
   + Implemented the binarization with a fixed threshold.<br>
   I use a single threshold to binarilize all 50 dimensions.<br>
   And the threshold I find best is:<br>
   Threshold = 0.18<br>
   F1 = 80.29
   
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
   
## Running
   Create a folder named `./glove.6B`, and move the `glove.6B.50d.txt` into it. Put the corpus files and `MEtag.java` and `MEtrain.java` and 
   and the required jar file in `./`. Set the `inmode` in the `WordEmbedding.py` to set the embedding methods.
   + "bin"      -> Binariazation method with fixed threshold
   + "bin_mean" -> Additional binariazation method with mean-value threshold
   + "cluster"  -> Cluster methods with K-means.<br>
   
   Then run the `WordEmbedding.py`. 
