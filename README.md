# "Bag of Words meets Bags of Popcorn" Kaggle tutorial

## What ?

My attempt at this [Kaggle tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial), to learn the Google's [Word2Vec](https://code.google.com/archive/p/word2vec/) package implementations for word representations as vectors (see *Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013.*)

## Who ?

Just me, with the provided help from Kaggle, like [this repo](https://github.com/wendykan/DeepLearningMovies).

## Why ?

Acquiring knowledge about *Natural Language Processing*, and more specifically to dive into the complex field of *Sentiment Analysis*.

## How ?

For a description of the code, see [code/README.md](https://github.com/NaasCraft/BOWtutorial/blob/master/code/README.md).

To run the Python code in this repo, you'll need the following packages :

+ [NumPy 1.9.2](http://www.numpy.org/)
+ [SciPy](http://www.scipy.org/)
+ [scikit-learn](http://scikit-learn.org/stable/)
+ [Natural Language Toolkit](http://www.nltk.org/) (nltk)
+ [Pandas 0.16.0](http://pandas.pydata.org/)
+ [BeautifulSoup 4](http://www.crummy.com/software/BeautifulSoup/)
+ [Cython](http://cython.org/)
+ [gensim](http://radimrehurek.com/gensim/index.html)

#### Command Line interface

The project is built to be used in command line. 
Here's an example :

+ __Objective__ (see [Results Report](https://github.com/NaasCraft/BOWtutorial/blob/master/code/submission/README.md) for details) :
    + build a RandomForest classifier (100 estimators), 
    + using Bag of Words features (5000), 
    + extracted from pre-processed data (Regex0, no stop words, no stemming)
    
+ __Part 1 : Preprocessing__
    + Run : `python main.py -v -p -pp`
    
        + "-v" argument will make the program output messages in the console
        + "-p" argument will force the pickling of pre-processed data
        + "-pp" argument will run the pre-processing script
        
    + You will be prompted something like this :
    ![Example console output 1](https://github.com/NaasCraft/BOWtutorial/blob/master/source/img/exampleCO_1.png)
        + Since we want the bag of word features, we don't need Word2Vec (so I typed "N")
        + You can check if the loaded database is the right shape (here, 25000 lines and 3 columns)
        + You can then choose your pre-processing parameters.
        
    + Now that you have pickled your pre-processed data, it is time for feature extraction, model fitting, and prediction submitting _(... I will split these steps in a further update, with a way of saving the data between each step. For now, the extracted features are way heavier than the data itself.)_ 
     
+ __Part 2 : Feature Extraction__
    + Run this code : `python main.py -v -fe -m -s`
    
        + "-fe" argument will run the feature extraction script
        + "-m" argument will run the model fitting script
        + "-s" argument will run the submission script
    
    + You will have to specify you don't want to use Word2Vec, enter the number of features (here 5000), and will then be prompted multiple information about the extracted features :
    ![Example console output 2](https://github.com/NaasCraft/BOWtutorial/blob/master/source/img/exampleCO_2.png)
        + You can then decide if you want to scale the computed features to approach a normally distributed data, which can be required for the model (see more [here](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html))

+ __Part 3 : Model fitting__
    + So here I typed "N". You will then be asked to choose a classifier; "rf" corresponds to RandomForest, "svm" to Support Vector Machine, and "knn" to k-Nearest Neighbors. Specify the number of estimators and the training will start :
    ![Example console output 3](https://github.com/NaasCraft/BOWtutorial/blob/master/source/img/exampleCO_3.png)
    
+ __Part 4 : Submitting the results__
    + The program then asks to give him the same parameters for pre-processing the test data. _(I will try to implement classes for data to carry those parameters and prevent from user errors)_
    
    + Some information is then showed, while the test data is pre-processed and its features extracted. The model then predicts the test labels and the output is saved in a .csv file.


## Left to do

- Compare with [__GloVe__](http://nlp.stanford.edu/projects/glove/)
- Try model learning with external corpus (e.g. [Latest Wikipedia dump](http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2))
- Phrases extraction  
- Hierarchical topic detection
- Other clustering algorithms