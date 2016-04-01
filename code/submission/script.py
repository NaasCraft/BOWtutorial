# coding=latin-1
#
# Last update : 31/03/2016
# Author : Naascraft
# Description : Kaggle tutorial on NLP with Word2Vec [SUBMISSION]

### Import module ###
import pickle as pkl
from time import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from main import debug

### Command Line Arguments ###
_verb = "-v" in sys.argv
_unpkl = "-u" in sys.argv

### Path variables ###
dataPath_ = "../source/data/"
outPath_ = "submission/"
fileDir_ = os.path.dirname(os.path.realpath('__file__'))
picklePath_ = os.path.join( fileDir_, "../pickles/" )

def run( model, modelID, verb=False, re_level_=0, sw_drop_=True, stem_=False, max_f=5000, vect=None, mode=False, wordModel=False, scale=False, dScaler=None ):
	# Test data retrieval
	test = pd.read_csv(dataPath_+"testData.tsv", header=0, delimiter="\t", quoting=3 )

	if verb: print ("\nTest dataset shape : " + str( test.shape ) )
	
	if not mode:
		import preprocess.script
		ppTest, _empt_ = preprocess.script.run( test, verbose=verb, re_level=re_level_, sw_drop=sw_drop_, stem=stem_ )
		debug(ppTest[0], "ppTest[0]")
		
		import sklmodels.script
		testFeatures, max_f, vect = sklmodels.script.getBoWf( ppTest, verbose=verb, vect=vect, m_f=max_f, default=True)
		
	else:
		import preprocess.script
		import w2v.script
		print( "Creating "+str(mode)+"-style feature vecs for test reviews" )
		
		clean_test_reviews = []
		for review in test["review"]:
			clean_test_reviews += [ preprocess.script.fullPPtoW( review, re_level=re_level_, \
						sw_drop=sw_drop_, stem=stem_, join_res=False ) ]
		
		testFeatures = w2v.script.loopFV( clean_test_reviews, wordModel, mode )
	
	if verb: print( "Example test feature (before scaling) : \n" + str( testFeatures[0] ) + "\n" )
	
	if scale:
		testFeatures = dScaler.transform( testFeatures )
		if verb: print( "Example test feature (after scaling) : \n" + str( testFeatures[0] ) + "\n" )
	
	result = model.predict(testFeatures)

	output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
	output.to_csv( outPath_ + "submission" + modelID + ".csv", index=False, quoting=3 )
	
	return output

'''
if __name__ == "__main__":
	# Data unpickling
	dataPath_ = "../" + dataPath_
	outPath_ = "../" + outPath_
	run( data, unpickle=_unpkl, verbose=_verb )
'''