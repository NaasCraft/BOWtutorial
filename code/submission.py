# coding=latin-1
#
# Last update : 05/04/2016
# Author : Naascraft
# Description : Kaggle tutorial on NLP with Word2Vec [SUBMISSION]

### Import module ###
import pickle as pkl
from time import time
import numpy as np
import pandas as pd
import sys
import os
import main

### Command Line Arguments ###
_verb = "-v" in sys.argv
_unpkl = "-u" in sys.argv
_default = "-d" in sys.argv

### Path variables ###
dataPath_, picklePath_, outPath_ = main.dataPath_, main.picklePath_, main.outPath_

### Debugging function ###
debug = main.debug

def run( model, modelID, verb=False, re_level=0, sw_drop=True, stem=False, max_f=5000, vect=None, mode=False, wordModel=False, scale=False, dScaler=None ):
	''' git description
+ __run__( model, modelID, verb=False, re_level=0, sw_drop=True, stem=False, max_f=5000, vect=None, mode=False, wordModel=False, scale=False, dScaler=None ) :
    + _does_ : 
        + Retrieves test data
        + Pre-processes it
        + Extract feature vectors according to "mode"
        + Predicts the test labels with "model"
        + Save the output as a Kaggle submission
    + _returns_ : Predicted output (as _DataFrame_)
    + _called by_ : `python main.py -s`
    + _calls_ : __pandas.read_csv__, __pandas.DataFrame__, __preprocess.run__, __preprocess.fullPPtoW__, __sklmodels.getBoWf__, __w2v.loopFV__
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
	'''
	
	# Test data retrieval
	test = pd.read_csv(dataPath_+"testData.tsv", header=0, delimiter="\t", quoting=3 )

	if verb: print ("\nTest dataset shape : " + str( test.shape ) )
	
	# Correct following if else statement with preprocess.run ability to give multiple values
	if not mode:
		import preprocess
		ppTest, _empt_ = preprocess.run( test, verbose=verb, re_level=re_level, sw_drop=sw_drop, stem=stem )
		
		import sklmodels
		testFeatures, max_f, vect = sklmodels.getBoWf( ppTest, verbose=verb, vect=vect, m_f=max_f, default=True)
		
	else:
		import preprocess
		import w2v
		print( "Creating "+str(mode)+"-style feature vecs for test reviews" )
		
		clean_test_reviews = []
		for review in test["review"]:
			clean_test_reviews += [ preprocess.fullPPtoW( review, re_level=re_level, \
						sw_drop=sw_drop, stem=stem, join_res=False ) ]
		
		testFeatures = w2v.loopFV( clean_test_reviews, wordModel, mode )
	
	if verb: print( "Example test feature (before scaling) : \n" + str( testFeatures[0] ) + "\n" )
	
	if scale:
		testFeatures = dScaler.transform( testFeatures )
		if verb: print( "Example test feature (after scaling) : \n" + str( testFeatures[0] ) + "\n" )
	
	result = model.predict(testFeatures)

	output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
	output.to_csv( outPath_ + "submission" + modelID + ".csv", index=False, quoting=3 )
	
	if verb: print( "Submission file saved as 'submission" + modelID + ".csv.")
	
	return output

'''
if __name__ == "__main__":
	# Data unpickling
	dataPath_ = "../" + dataPath_
	outPath_ = "../" + outPath_
	run( data, unpickle=_unpkl, verbose=_verb )
'''