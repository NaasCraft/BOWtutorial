# coding=latin-1
#
# Last update : 31/03/2016
# Author : Naascraft
# Description : Kaggle tutorial on NLP with Word2Vec

### Import module ###
import pickle as pkl
from time import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
from sklearn.ensemble import RandomForestClassifier

### Command Line Arguments ###
_verb = "-v" in sys.argv
_unpkl = "-u" in sys.argv

### Path variables ###
dataPath_ = "../source/data/"
fileDir_ = os.path.dirname(os.path.realpath('__file__'))
picklePath_ = os.path.join( fileDir_, "../pickles/" )

def run( model, modelID, verb=False, re_level_=0, sw_drop_=True, stem_=False ):
	# Test data retrieval
	test = pd.read_csv(dataPath_+"testData.tsv", header=0, delimiter="\t", quoting=3 )

	if verb: print ("\nTest dataset shape : " + str( test.shape ) )
	
	import preprocess.script
	ppTest = preprocess.script.run( test, verbose=verb, re_level=re_level_, sw_drop=sw_drop_, stem=stem_ )
	
	import sklmodels.script
	testFeatures = sklmodels.script.getBoWf( ppTest, verbose=verb )

	result = model.predict(testFeatures)

	output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
	output.to_csv( "submission"+modelID+".csv", index=False, quoting=3 )
	
	return output

if __name__ == "__main__":
	# Data unpickling
	dataPath_ = "../"+dataPath_
	run( data, unpickle=_unpkl, verbose=_verb )