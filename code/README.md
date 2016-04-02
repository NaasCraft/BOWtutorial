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
    + _imported in_ : `preprocess.py`, `sklmodels.py`, `submission.py`, `w2v.py`
    + _calls_ : Nothing
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
    + _does_ : 
    + _called by_ : 
    + _calls_ : 
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | text | Text to be treated |
| _list_ of _list_ of _string_ | lSubs | List of Regex substitution pairs |

+ __reTreatment__( text, level=0 ) :
    + _does_ : 
    + _called by_ : 
    + _calls_ : 
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | text | Text to be treated |
| _int_ | level | Level of Regex treatment (0-3) |

+ __rmStopword__( words ) :
    + _does_ : 
    + _called by_ : 
    + _calls_ : 
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | words | List of tokens to be treated |

+ __pStem__( words ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | words | List of tokens to be treated |

+ __fullPPtoW__( review, re_level, sw_drop, stem, join_res=True ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | review | Review to be pre-processed |
| _int_ | re_level | Level of Regex treatment (0-3) |
| _boolean_ | sw_drop | Should drop stop words |
| _boolean_ | stem | Should apply Porter Stemming |
| _boolean_ | join_res | Should return result as string (else as list of words) |

+ __fullPPtoS__( review, re_level, sw_drop, stem, tk=tokenizer_ ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | review | Review to be pre-processed into sentences |
| _int_ | re_level | Level of Regex treatment (0-3) |
| _boolean_ | sw_drop | Should drop stop words |
| _boolean_ | stem | Should apply Porter Stemming |
| _Tokenizer_ (from __nltk__) | tk | Tokenizer to split into sentences |

+ __run__( data, verbose=False, re_level=0, sw_drop=True, stem=False, asW2V=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
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
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | train |  |

+ __buildRF__( features, label, n_est=100, verbose=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | features |  |
|  | label |  |
|  | n_est |  |
| _boolean_ | verbose | Controls console outputs |

+ __buildSVM__( features, label, verbose=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | features |  |
|  | label |  |
|  | verbose |  |

+ __buildKNN__( features, label, verbose=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | features |  |
|  | label |  |
|  | verbose |  |

+ __buildModel__( features, label, mode="rf", verbose=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | features |  |
|  | label |  |
|  | mode |  |
|  | verbose |  |

+ __getBoWf__( data=[], unpickle=False, verbose=False, m_f=5000, default=False, vect=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | data |  |
|  | unpickle |  |
|  | verbose |  |
|  | m_f |  |
|  | default |  |
|  | vect |  |


## `submission.py`

#### Command Line Arguments

+ __-v__ : "Verbose", controls console outputs.
+ __-u__ : "Unpickling", controls whether to load the data or not, from `pickles/` folder.
+ __-d__ : "Default", runs with default parameters.

#### Functions

+ __run__( model, modelID, verb=False, re_level=0, sw_drop=True, stem=False, max_f=5000, vect=None, mode=False, wordModel=False, scale=False, dScaler=None ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | model |  |
|  | modelID |  |
|  | verb |  |
|  | re_level |  |
|  | sw_drop |  |
|  | stem |  |
|  | max_f |  |
|  | vect |  |
|  | mode |  |
|  | wordModel |  |
|  | scale |  |
|  | dScaler |  |


## `w2v.py`

#### Command Line Arguments

+ __-v__ : "Verbose", controls console outputs.
+ __-u__ : "Unpickling", controls whether to load the data or not, from `pickles/` folder.
+ __-d__ : "Default", runs with default parameters.
+ __-s__ : "Save", controls whether to save the language model, into `models/` folder.

#### Functions

+ __makeFeatureVec__( words, model, num_features ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | words |  |
|  | model |  |
|  | num_features |  |

+ __showClusters__ ( map, n ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | map |  |
|  | n |  |

+ __kMeansFit__ ( data, num_clusters ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | data |  |
|  | num_clusters |  |

+ __createBagOfCentroids__( wordlist, word_centroid_map ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | wordlist |  |
|  | word_centroid_map |  |

+ __loopFV__( reviews, model, mode="avg", dump=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | reviews |  |
|  | model |  |
|  | mode |  |
|  | dump |  |

+ __load__( model_name, rd=True ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | model_name |  |
|  | rd |  |

+ __notMatch__( model, words, verbose ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | model |  |
|  | words |  |
|  | verbose |  |

+ __modelTesting__( model ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | model |  |

+ run( sentences, save=False, default=False, verbose=False, ready=True, lf=False, loadname="" ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  | sentences |  |
|  | save |  |
|  | default |  |
|  | verbose |  |
|  | ready |  |
|  | lf |  |
|  | loadname |  |