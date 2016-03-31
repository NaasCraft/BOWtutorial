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
dataPath_ = "../../source/data/"
fileDir_ = os.path.dirname(os.path.realpath('__file__'))
picklePath_ = os.path.join( fileDir_, "../pickles/" )

def buildRF( features, label, n_est=100, verbose=False ):
	if verbose: print( "Training the random forest model with " + str(n_est) + " estimators...\n" )
	t = time()
	forest = RandomForestClassifier(n_estimators = n_est) 
	forest = forest.fit( features, label )
	
	if verbose: print( "Completed in " + str( time()-t ) + " seconds.\n" )
	return forest
	
def getBoWf( data=[], unpickle=False, verbose=False ):
	print( "You required bag of words learning. Enter the following parameters : \n" )
	in_max_features = int( raw_input( "Maximum features : " ) )
	
	if unpickle:
		t = time()
		data = pkl.load( open( picklePath_+"ppTrainData.pkl", "rb" ) )
		print("\nPreprocessed data unpickling successfully completed in " + str( time()-t ) + " seconds.")
	
	if verbose: print( "\nCreating the bag of words...\n")
	
	t = time()
	vectorizer = CountVectorizer(analyzer = "word", \
						tokenizer = None, \
						preprocessor = None, \
						stop_words = None, \
						max_features = in_max_features) 
	
	bowFeatures = vectorizer.fit_transform(data)
	bowFeatures = bowFeatures.toarray()
	
	if verbose: 
		print("Bag of words features extracted in " + str( time()-t ) + " seconds.\n" + \
			"Shape of features dataset : " + str( bowFeatures.shape ) + "\n" + \
			"First 20 most common words in the learned vocabulary : \n" )
		vocab = vectorizer.get_feature_names()
		dist = np.sum( bowFeatures, axis=0 )
		order = np.rec.fromarrays( [dist, vocab] )
		order.sort()
		order = order[::-1]
		j = 0
		
		for tag, count in order:
			if j < 20:
				print( str(tag) + " : " + str(count) )
				j += 1
			else:
				return bowFeatures
	else:
		return bowFeatures
	
if __name__ == "__main__":
	# Data unpickling
	getBoWf( data, unpickle=_unpkl, verbose=_verb )