# coding=latin-1
#
# Last update : 02/04/2016
# Author : Naascraft
# Description : Kaggle tutorial on NLP with Word2Vec [SKLMODELS]

### Import module ###
import pickle as pkl
from time import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
import main

### Command Line Arguments ###
_verb = "-v" in sys.argv
_unpkl = "-u" in sys.argv
_default = "-d" in sys.argv

### Path variables ###
dataPath_, picklePath_ = main.dataPath_, main.picklePath_

### Debugging function ###
debug = main.debug

def dataScaler( train ):
	from sklearn.preprocessing import StandardScaler
	
	scaler = StandardScaler()
	scaler.fit( train )
	
	return scaler

def buildRF( features, label, n_est=100, verbose=False ):
	from sklearn.ensemble import RandomForestClassifier
	
	if verbose: print( "Training the random forest model with " + str(n_est) + " estimators...\n" )
	t = time()
	forest =RandomForestClassifier(n_estimators = n_est) 
	forest = forest.fit( features, label )
	
	if verbose: print( "Completed in " + str( time()-t ) + " seconds.\n" )
	return forest
	
''' NOT WORKING with scikit-learn version 0.17 (need 0.18)
def buildMLP( features, label, verbose=False ):
	from sklearn.neural_network import MLPClassifier
	
	if verbose: print( "Training the multi-layer perceptron model...\n" )
	t = time()
	mlp = RandomForestClassifier(algorithm='l-bfgs', alpha=1e-3, hidden_layer_sizes=(5, 2), random_state=1) 
	mlp = mlp.fit( features, label )
	
	if verbose: print( "Completed in " + str( time()-t ) + " seconds.\n" )
	return mlp
'''

def buildSVM( features, label, verbose=False ):
	from sklearn.svm import SVC
	
	if verbose: print( "Training the SVM model with gaussian kernel...\n" )
	t = time()
	svc = SVC(kernel='rbf', gamma='auto') 
	svc = svc.fit( features, label )
	
	if verbose: print( "Completed in " + str( time()-t ) + " seconds.\n" )
	return svc

def buildKNN( features, label, verbose=False ):
	from sklearn.neighbors import KNeighborsClassifier
	
	if verbose: print( "Training the K-NN model with k=10...\n" )
	t = time()
	clf = KNeighborsClassifier(n_neighbors=10) 
	clf = clf.fit( features, label )
	
	if verbose: print( "Completed in " + str( time()-t ) + " seconds.\n" )
	return clf

def buildModel( features, label, mode="rf", verbose=False ):
	if mode == "rf":
		in_n_est = int( raw_input( "Enter number of estimators for the Random Forest classifier : " ) )
		return buildRF( features, label, in_n_est, verbose )
		
	#elif mode == "mlp":
	#	return buildMLP( features, label, verbose )
		
	elif mode == "svm":
		return buildSVM( features, label, verbose )
		
	elif mode == "knn":
		return buildKNN( features, label, verbose )

def getBoWf( data=[], unpickle=False, verbose=False, m_f=5000, default=False, vect=False ):
	if not default:
		print( "You required bag of words learning. Enter the following parameters : \n" )
		m_f = int( raw_input( "Maximum features : " ) )
	
	if unpickle:
		t = time()
		data = pkl.load( open( picklePath_+"ppTrainData.pkl", "rb" ) )
		print("\nPreprocessed data unpickling successfully completed in " + str( time()-t ) + " seconds.")
	
	if not vect:
		if verbose: print( "\nCreating the bag of words...\n")
		
		t = time()
		vectorizer = CountVectorizer(analyzer = "word", \
							tokenizer = None, \
							preprocessor = None, \
							stop_words = None, \
							max_features = m_f) 
		
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
					return bowFeatures, m_f, vectorizer
		else:
			return bowFeatures, m_f, vectorizer
	
	else:
		if verbose: print( "\nCreating the bag of words features for test data...\n")
		bowFeatures = vect.transform(data)
		bowFeatures = bowFeatures.toarray()
		
		return bowFeatures, m_f, vect
	
if __name__ == "__main__":
	# Bag of word extraction as default
	getBoWf( data, unpickle=_unpkl, verbose=_verb, default=_default )