# Full code description

## `main.py`

#### Command Line Arguments

+ __-v__ : "Verbose", controls console outputs.
+ __-p__ : "Pickling", controls whether to save the data or not, into `pickles/` folder.
+ __-h__ : "Help", if present, simply show a description of command line arguments.
+ __-d__ : "Default", runs with default parameters.
+ __-nd__ : "No Data", runs without loading data. Used for language model testing.
+ __-pp__ : "Pre-processing", runs the data pre-processing.
+ __-fe__ : "Feature extraction", runs the feature extraction (according to user inputs).
+ __-m__ : "Modelling", runs the model training (according to user inputs).
+ __-s__ : "Submit", runs the predicting and submission writing.


#### Functions

+ __debug__( var, repr ) :
    + _does_ : Print out "debug repr" followed by (_var_) value
    + _returns_ : _Nothing_
    + _imported in_ : `preprocess.py`, `sklmodels.py`, `submission.py`, `w2v.py`
    + _calls_ : _Nothing_
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| any | var | Variable to debug |
| _string_ | repr | Variable representation |


## `preprocess.py`

#### Command Line Arguments

+ __-v__ : "Verbose", controls console outputs.
+ __-d__ : "Default", runs with default parameters.

#### Functions

+ __reSub__( text, lSubs ) :
    + _does_ : Performs a "lSubs" list of regular expression substitutions in the "text" string parameter
    + _returns_ : Treated "text" (as _string_)
    + _called by_ : __reTreatment__
    + _calls_ : __copy.copy__, __re.sub__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | text | Text to be treated |
| _list_ of _list_ of _string_ | lSubs | List of Regex substitution pairs |

+ __reTreatment__( text, level=0 ) :
    + _does_ : Performs a Regex treatment on "text" string as defined by "level" parameter (0 keeps _only alphabetica_l chars, 1 adds _numbers_ as "num", 2 adds _punctuation_ as "punct", 3 adds _emoticons_ as "emo")
    + _returns_ : Treated text (as _string_)
    + _called by_ : __fullPPtoW__
    + _calls_ : __reSub__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | text | Text to be treated |
| _int_ | level | Level of Regex treatment (0-3) |

+ __rmStopword__( words ) :
    + _does_ : Removes stopwords from the given "words" list of tokens
    + _returns_ : Treated list of words (as _list_)
    + _called by_ : __fullPPtoW__
    + _calls_ : __nltk.corpus.stopwords__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | words | List of tokens to be treated |

+ __pStem__( words ) : Applies the Porter Stemming algorithm to the given list of tokens
    + _does_ : Applies the Porter Stemming algorithm to the given "words" list of tokens
    + _returns_ : Treated list of words (as _list_)
    + _called by_ : __fullPPtoW__
    + _calls_ : __nltk.stem.porter.PorterStemmer__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | words | List of tokens to be treated |

+ __fullPPtoW__( review, re_level, sw_drop, stem, join_res=True ) :
    + _does_ : Computes a full "review" string pre-processing, according to ("re_level", "sw_drop", "stem") parameters
    + _returns_ : Treated "review" (as _list_ or _string_ depending on "join_res")
    + _called by_ : __fullPPtoS__, __run__, __submission.run__
    + _calls_ : __reTreatment__, __rmStopWords__, __pStem__, __bs4.BeautifulSoup__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | review | Review to be pre-processed |
| _int_ | re_level | Level of Regex treatment (0-3) |
| _boolean_ | sw_drop | Should drop stop words |
| _boolean_ | stem | Should apply Porter Stemming |
| _boolean_ | join_res | Should return result as string (else as list of words) |

+ __fullPPtoS__( review, re_level, sw_drop, stem, tk=tokenizer_ ) :
    + _does_ : Computes a full "review" string pre-processing into sentences split by "tk", according to ("re_level", "sw_drop", "stem") parameters
    + _returns_ : Treated review into list of sentences (as _list_)
    + _called by_ : __run__
    + _calls_ : __fullPPtoW__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | review | Review to be pre-processed into sentences |
| _int_ | re_level | Level of Regex treatment (0-3) |
| _boolean_ | sw_drop | Should drop stop words |
| _boolean_ | stem | Should apply Porter Stemming |
| _Tokenizer_ (from __nltk__) | tk | Tokenizer to split into sentences |

+ __run__( data, verbose=False, re_level=0, sw_drop=True, stem=False, asW2V=False ) :
    + _does_ : Full "data" pre-processing according to given parameters
    + _returns_ : (__2__ values)
        + if "asW2V" : Treated data as sentences (as _list_), ... as words (as _list_)
        + else : Treated data as words (as _list_), empty list
    + _called by_ : `python main.py -pp`, __submission.run__
    + _calls_ : __time.time__, __fullPPtoS__, __fullPPtoW__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _ndarray_ (from __numpy__) | data | Dataset to be pre-processed |
| _boolean_ | verbose | Controls console outputs |
| _int_ | re_level | Level of Regex treatment (0-3) |
| _boolean_ | sw_drop | Should drop stop words |
| _boolean_ | stem | Should apply Porter Stemming |
| _boolean_ | asW2V | Should use sentences (else just words) |


## `sklmodels.py`

#### Command Line Arguments

+ __-v__ : "Verbose", controls console outputs.
+ __-u__ : "Unpickling", controls whether to load the data or not, from `pickles/` folder.
+ __-d__ : "Default", runs with default parameters.


#### Functions

+ __dataScaler__( train ) :
    + _does_ : 
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | train | List of train features to fit the scaler |

+ __buildRF__( features, label, n_est=100, verbose=False ) :
    + _does_ : 
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | features | List of train features to fit the model |
| _list_ of _int_ | label | List of associated labels |
| _int_ | n_est | Number of estimators for the RF |
| _boolean_ | verbose | Controls console outputs |

+ __buildSVM__( features, label, verbose=False ) :
    + _does_ : 
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | features | List of train features to fit the model |
| _list_ of _int_ | label | List of associated labels |
| _boolean_ | verbose | Controls console outputs |

+ __buildKNN__( features, label, verbose=False ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | features | List of train features to fit the model |
| _list_ of _int_ | label | List of associated labels |
| _boolean_ | verbose | Controls console outputs |

+ __buildModel__( features, label, mode="rf", verbose=False ) :
    + _does_ : 
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | features | List of train features to fit the model |
| _list_ of _int_ | label | List of associated labels |
| _string_ | mode | Defines which model to train |
| _boolean_ | verbose | Controls console outputs |

+ __getBoWf__( data=[], unpickle=False, verbose=False, m_f=5000, default=False, vect=False ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | data | List of pre-processed reviews |
| _boolean_ | unpickle | Loads train data (else uses given data) |
| _boolean_ | verbose | Controls console outputs |
| _int_ | m_f | Number of maximum features for the Bag of Words |
| _boolean_ | default | Runs with default parameters (else asks user input) |
| _CountVectorizer_ (from __sklearn__) | vect | Saved vectorizer to transform test data |


## `submission.py`

#### Command Line Arguments

+ __-v__ : "Verbose", controls console outputs.
+ __-u__ : "Unpickling", controls whether to load the data or not, from `pickles/` folder.
+ __-d__ : "Default", runs with default parameters.

#### Functions

+ __run__( model, modelID, verb=False, re_level=0, sw_drop=True, stem=False, max_f=5000, vect=None, mode=False, wordModel=False, scale=False, dScaler=None ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _classifier_ (from __sklearn__) | model | Trained model for prediction |
| _string_ | modelID | Describes model and feature extraction mode for output |
| _boolean_ | verb | Controls console outputs |
| _int_ | re_level | Level of Regex treatment (0-3) |
| _boolean_ | sw_drop | Should drop stop words |
| _boolean_ | stem | Should apply Porter Stemming |
| _int_ | max_f | Number of maximum features for the Bag of Words |
| _CountVectorizer_ (from __sklearn__) | vect | Saved vectorizer to transform test data |
| _string_ | mode | Feature extraction mode (None for BoW, "avg" or "cluster") |
| _W2VModel_ (from __gensim__) | wordModel | Trained word vector representation model |
| _boolean_ | scale | Apply data scaling |
| _StandardScaler_ (from __sklearn__) | dScaler | Fitted data scaler |


## `w2v.py`

#### Command Line Arguments

+ __-v__ : "Verbose", controls console outputs.
+ __-u__ : "Unpickling", controls whether to load the data or not, from `pickles/` folder.
+ __-d__ : "Default", runs with default parameters.
+ __-s__ : "Save", controls whether to save the language model, into `models/` folder.

#### Functions

+ __makeFeatureVec__( words, model, num_features ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | words | List of words to extract features from |
| _W2VModel_ (from __gensim__) | model | Trained word vector representation model |
| _int_ | num_features | Size of word vector representations |

+ __showClusters__ ( map, n ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _dict_ | map | Map of word cluster centroids |
| _int_ | n | Number of first clusters to show |

+ __kMeansFit__ ( data, num_clusters ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _ndarray_ (from __numpy__) | data | Vocabulary learned by the W2V model |
| _int_ | num_clusters | Number of clusters to compute by KMeans() |

+ __createBagOfCentroids__( wordlist, word_centroid_map ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | wordlist | List of words to extract features from |
| _dict_ | word_centroid_map | Map of word cluster centroids |

+ __loopFV__( reviews, model, mode="avg", dump=False ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _list_ of _string_ | reviews | List of reviews as lists of words |
| _W2VModel_ (from __gensim__) | model | Trained word vector representation model |
| _string_ | mode | Feature extraction mode ("avg" or "cluster") |
| _boolean_ | dump | Computes clusters and pickles the word cluster centroids map (else loads it) |

+ __load__( model_name, rd=True ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | model_name | Name of model to load |
| _boolean_ | rd | Trims unneeded model memory but prevents model to be trained again |

+ __notMatch__( model, words, verbose ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _W2VModel_ (from __gensim__) | model | Trained word vector representation model |
| _list_ of _string_ | words | List of words to test |
| _boolean_ | verbose | Controls console outputs |

+ __modelTesting__( model ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _W2VModel_ (from __gensim__) | model | Trained word vector representation model |

+ run( sentences, save=False, default=False, verbose=False, ready=True, lf=False, loadname="" ) :
    + _does_ :
    + _returns_ :  (as __)
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _list_ of _string_ | sentences | List of sentences as lists of words to train the model |
| _boolean_ | save | Saves model once trained |
| _boolean_ | default | Runs with default parameters (else asks user input) |
| _boolean_ | verbose | Controls console outputs |
| _boolean_ | ready | Trims unneeded model memory but prevents model to be trained again |
| _boolean_ | lf | Loads _loadname_ trained model |
| _string_ | loadname | Name of model to load |