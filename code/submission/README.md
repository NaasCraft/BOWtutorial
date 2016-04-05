# Results Report

## Methods

For this tutorial, I have implemented :

+ 3 types of classifiers :
    + [Random Forest](http://scikit-learn.org/stable/modules/ensemble.html#forest)
    + [K-Nearest Neighbors](http://scikit-learn.org/stable/modules/neighbors.html)
    + [SVM](http://scikit-learn.org/stable/modules/svm.html#classification)

+ 3 methods of preprocessing :
    + Pre-treatment of raw text through regular expressions with multiple levels :
        + __0__ : Only keeps alphabetical characters
        + __1__ : Adds numbers represented as NUM
        + __2__ : Adds punctuation characters as PUNCT
        + __3__ : Adds emoticons as EMO ([see here](http://sentiment.christopherpotts.net/tokenizing.html#emoticons))
        
    + Removal of english stop words as defined in the [NLTK corpus](http://www.nltk.org/book/ch02.html#code-unusual)
    + [Porter Stemming Algorithm](http://www.nltk.org/howto/stem.html)
    
+ 3 feature extraction techniques :
    + Simple bag of words features (see [Kaggle tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)) (__5000__ features)
    + With Word2Vec vector representation of words (see [tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors)) (__300__ features) :
        + with averaging words over a review
        + with clustering of learned vocabulary used to extract "ba of words"-like features
        
## Results : Prediction accuracy scores

### Random Forest

#### Parameters : (number of estimators = __100__)

|  | Bag of Words | Averaged Word Vectors | Bag of Centroids |
| :---: | ---: | ---: | ---: |
| Regex3, noSW, noStem | 0.84416 | 0.82960 | __0.84688__ |
| Regex3, noSW, Stem | 0.84348 | 0.78080 |  |
| Regex0, noSW, noStem | __0.84628__ | __0.83388__ | 0.84304 |
| Regex0, SW, noStem |  |  | 0.58096 |

### K Nearest Neighbors

#### Parameters : (k = __3__)

|  | Bag of Words | Averaged Word Vectors | Bag of Centroids |
| :---: | ---: | ---: | ---: |
| Regex3, noSW, noStem | 0.57980 | 0.76552 |  |
| Regex0, SW, noStem |  |  | 0.53736 |

#### Parameters : (k = __10__)

|  | Bag of Words | Averaged Word Vectors | Bag of Centroids |
| :---: | ---: | ---: | ---: |
| Regex3, noSW, noStem | 0.58264 |  | 0.63996 |

### Support Vector Machine

#### Parameters : (kernel = __gaussian__), (C = __1.0__)

|  | Bag of Words | Averaged Word Vectors | Bag of Centroids |
| :---: | ---: | ---: | ---: |
| Regex1, noSW, noStem | 0.84544 |  |  |

### Aggregating models - an example

Soft Voting classifier combining :

+ (weight = 2) RandomForest with 100 estimators
+ (weight = 1) 5-Nearest Neighbors
+ (weight = 2) Gaussian kernel SVM

_Preprocessing mode : Regex0, noSW, noStem_

|  | RandomForest | 5-NN | SVM | SoftVoting |
| :---: | ---: | ---: | ---: | ---: |
| Bag of Words 2000 | 0.83848 | 0.61544 | 0.85796 | 0.86184 |
| Bag of Words 5000 |  |  |  |  |