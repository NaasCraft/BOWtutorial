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
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |


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
|  |  |  |

+ __reTreatment__( text, level=0 ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __rmStopword__( words ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __pStem__( words ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __fullPPtoW__( review, re_level, sw_drop, stem, join_res=True ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __fullPPtoS__( review, re_level, sw_drop, stem, tk=tokenizer_ ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __run__( data, verbose=False, re_level=0, sw_drop=True, stem=False, asW2V=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |


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
|  |  |  |

+ __buildRF__( features, label, n_est=100, verbose=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __buildSVM__( features, label, verbose=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __buildKNN__( features, label, verbose=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __buildModel__( features, label, mode="rf", verbose=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __getBoWf__( data=[], unpickle=False, verbose=False, m_f=5000, default=False, vect=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |




## `submission.py`

#### Command Line Arguments

+ __-v__ : "Verbose", controls console outputs.
+ __-u__ : "Unpickling", controls whether to load the data or not, from `pickles/` folder.
+ __-d__ : "Default", runs with default parameters.


#### Functions

+ __run__( model, modelID, verb=False, re_level_=0, sw_drop_=True, stem_=False, max_f=5000, vect=None, mode=False, wordModel=False, scale=False, dScaler=None ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |



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
|  |  |  |

+ __showClusters__ ( map, n ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __kMeansFit__ ( data, num_clusters ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __createBagOfCentroids__( wordlist, word_centroid_map ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __loopFV__( reviews, model, mode="avg", dump=False ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __load__( model_name, rd=True ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __notMatch__( model, words, verbose ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ __modelTesting__( model ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |

+ run( sentences, save=False, default=False, verbose=False, ready=True, lf=False, loadname="" ) :
    + _does_ :
    + _called by_ :
    + _calls_ :
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
|  |  |  |