# "Bag of Words meets Bags of Popcorn" Kaggle tutorial

## What ?

My attempt at this [Kaggle tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial), to learn the Google's [Word2Vec](https://code.google.com/archive/p/word2vec/) package implementations for word representations as vectors (see *Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013.*)

## Who ?

Just me, with the provided help from Kaggle, like [this repo](https://github.com/wendykan/DeepLearningMovies).

## Why ?

Acquiring knowledge about *Natural Language Processing*, and more specifically to dive into the complex field of *Sentiment Analysis*.

## How ?

To run the Python code in this repo, you'll need the following packages :

+ [NumPy 1.9.2](http://www.numpy.org/)
+ [SciPy](http://www.scipy.org/)
+ [scikit-learn](http://scikit-learn.org/stable/)
+ [Natural Language Toolkit](http://www.nltk.org/) (nltk)
+ [Pandas 0.16.0](http://pandas.pydata.org/)
+ [BeautifulSoup 4](http://www.crummy.com/software/BeautifulSoup/)
+ [Cython](http://cython.org/)
+ [gensim](http://radimrehurek.com/gensim/index.html)

### [TODO]

- [x] [Part 1](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words) : For Beginners - Bag of Words
    - [x] _\[extension\]_ : Explore regular expressions for preprocessing
    - [x] _\[extension\]_ : Implement stemming
    - [ ] _\[extension\]_ : Build different classifiers (other than RF)
    
- [x] [Part 2](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors) : Word Vectors
    - [ ] _\[extension\]_ : Compare with [__GloVe__](http://nlp.stanford.edu/projects/glove/)
    - [ ] _\[extension\]_ : Try model learning with external corpus (e.g. [Latest Wikipedia dump](http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2))
    - [ ] _\[extension\]_ : Phrases extraction
    
- [ ] [Part 3](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors) : More Fun With Word Vectors
    - [ ] _\[extension\]_ : Hierarchical topic detection
    - [ ] _\[extension\]_ : Other clustering algorithms
