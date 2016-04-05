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
	''' git description
+ __dataScaler__( train ) :
    + _does_ : Fits a standard feature scaler on "train" data
    + _returns_ : Fitted scaler (as _StandardScaler_)
    + _called by_ : `python main.py -skl -f`
    + _calls_ : __sklearn.preprocessing.StandardScaler__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | train | List of train features to fit the scaler |
	'''
	
	from sklearn.preprocessing import StandardScaler
	
	scaler = StandardScaler()
	scaler.fit( train )
	
	return scaler

def buildRF( features, label, n_est=100, verbose=False ):
	''' git description
+ __buildRF__( features, label, n_est=100, verbose=False ) :
    + _does_ : Fits a RandomForestClassifier with "n_est" estimators, on ("features", "label") data
    + _returns_ : Fitted classifier (as _RandomForestClassifier_)
    + _called by_ : __buildModel__
    + _calls_ : __sklearn.ensemble.RandomForestClassifier__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | features | List of train features to fit the model |
| _list_ of _int_ | label | List of associated labels |
| _int_ | n_est | Number of estimators for the RF |
| _boolean_ | verbose | Controls console outputs |
	'''
	
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



def buildSVM( features, label, kernel="rbf", verbose=False ):
	''' git description
+ __buildSVM__( features, label, verbose=False ) :
    + _does_ : Fits a SVM classifier with "kernel" kernel, on ("features", "label") data
    + _returns_ : Fitted classifier (as _SVC_)
    + _called by_ : __buildModel__
    + _calls_ : __sklearn.svm.SVC__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | features | List of train features to fit the model |
| _list_ of _int_ | label | List of associated labels |
| _string_ | kernel | Kernel type ("linear", "poly", "rbf" or "sigmoid") |
| _boolean_ | verbose | Controls console outputs |
	'''
	
	from sklearn.svm import SVC
	
	if verbose: print( "Training the SVM model with '" + str(kernel) + "' kernel...\n" )
	t = time()
	svc = SVC(kernel=kernel, gamma='auto') 
	svc = svc.fit( features, label )
	
	if verbose: print( "Completed in " + str( time()-t ) + " seconds.\n" )
	return svc

def buildKNN( features, label, n_neighbors=3, verbose=False ):
	''' git description
+ __buildKNN__( features, label, n_neighbors=3, verbose=False ) :
    + _does_ : Fits a k-NN classifier with k = "n_neighbors", on ("features", "label") data
    + _returns_ : Fitted classifier (as _KNeighborsClassifier_)
    + _called by_ : __buildModel__
    + _calls_ : __sklearn.neighbors.KNeighborsClassifier__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | features | List of train features to fit the model |
| _list_ of _int_ | label | List of associated labels |
| _int_ | n_neighbors | Number of neighbors in the vote count (_k_) |
| _boolean_ | verbose | Controls console outputs |
	'''
	
	from sklearn.neighbors import KNeighborsClassifier
	
	if verbose: print( "Training the K-NN model with k=" + str(n_neighbors) + "...\n" )
	t = time()
	clf = KNeighborsClassifier(n_neighbors=3) 
	clf = clf.fit( features, label )
	
	if verbose: print( "Completed in " + str( time()-t ) + " seconds.\n" )
	return clf

def buildVoting( features, label, params, verbose=False ):
	''' git description
+ __buildVoting__( features, label, params, verbose=False ) :
    + _does_ : Fits a voting classifier aggregating a RF a SVM and a KNN classifiers, on ("features", "label") data
    + _returns_ : Fitted model (as _?_, has to be a _sklearn_ classifier though)
    + _called by_ : __buildModel__
    + _calls_ : __sklearn.ensemble.RandomForestClassifier__, __sklearn.svm.SVC__, __sklearn.neighbors.KNeighborsClassifier__, __sklearn.ensemble.VotingClassifier__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | features | List of train features to fit the model |
| _list_ of _int_ | label | List of associated labels |
| _list_ | params | List of model parameters [n_estimators, n_neighors, kernel] |
| _boolean_ | verbose | Controls console outputs |
	'''
	
	from sklearn.ensemble import RandomForestClassifier, VotingClassifier
	from sklearn.svm import SVC
	from sklearn.neighbors import KNeighborsClassifier
	
	clf1 = RandomForestClassifier( n_estimators = params[0] )
	clf2 = KNeighborsClassifier( n_neighbors = params[1] )
	clf3 = SVC( kernel = params[2], probability = True )
	
	t= time()
	if verbose: print( "Training a voting classifier from following models : \n" + \
				"  - Random Forest (" + str(params[0]) +" estimators) - weight = 2 \n" + \
				"  - " + str(params[1]) + "-Nearest Neighbors - weight = 1 \n" + \
				"  - SVM ('" + str(params[2]) + "' kernel) - weight = 2 \n\n" + \
				"Please wait...\n" )
	agg_clf = VotingClassifier( estimators=[ ('rf', clf1), ('knn', clf2), ('svm', clf3) ], voting='soft', weights=[2,1,2] )
	agg_clf.fit( features, label )
	
	if verbose: print( "Completed in " + str( time()-t ) + " seconds.\n" )
	return agg_clf

def buildModel( features, label, mode="rf", verbose=False ):
	''' git description
+ __buildModel__( features, label, mode="rf", verbose=False ) :
    + _does_ : Fits a classifier given by "mode" (or more if mode="agg"), on ("features", "label") data
    + _returns_ : Fitted model (as _?_, has to be a _sklearn_ classifier though)
    + _called by_ : `python main.py -fe -m`
    + _calls_ : __buildRF__, __buildSVM__, __buildKNN__, __buildVoting__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ | features | List of train features to fit the model |
| _list_ of _int_ | label | List of associated labels |
| _string_ | mode | Defines which model to train |
| _boolean_ | verbose | Controls console outputs |
	'''
	
	df = raw_input( "Default parameters for models ? (Y/N) :" ) == "Y"
	if df:
		params = [100, 5, "rbf"]
	else:
		n_est = int( raw_input( "Number of estimators for RF : ") )
		n_neigh = int( raw_input( "Number of neighbors votes for k-NN : ") )
		kern = raw_input( "Kernel type for SVM (rbf, poly, linear, sigmoid) : ")
		params = [n_est, n_neigh, kern]
	
	if mode == "rf":
		return buildRF( features, label, params[0], verbose )
		
	#elif mode == "mlp":
	#	return buildMLP( features, label, verbose )
		
	elif mode == "svm":
		return buildSVM( features, label, params[2], verbose )
		
	elif mode == "knn":
		return buildKNN( features, label, params[1], verbose )
		
	elif mode == "agg":
		return buildVoting( features, label, params, verbose )


def getBoWf( data=[], unpickle=False, verbose=False, m_f=5000, default=False, vect=False ):
	''' git description
+ __getBoWf__( data=[], unpickle=False, verbose=False, m_f=5000, default=False, vect=False ) :
    + _does_ : Extract bag of words "m_f" number of features from "data"
    + _returns_ : Extracted features (as _ndarray_), maximum features (as _int_), BoW feature extractor (as _CountVectorizer_)
    + _called by_ : `python main.py -skl`, __submission.run__
    + _calls_ : __sklearn.feature_extraction.text.CountVectorizer__, __pickle.load__, __numpy.sum__, __numpy.rec.fromarrays__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | data | List of pre-processed reviews |
| _boolean_ | unpickle | Loads train data (else uses given data) |
| _boolean_ | verbose | Controls console outputs |
| _int_ | m_f | Number of maximum features for the Bag of Words |
| _boolean_ | default | Runs with default parameters (else asks user input) |
| _CountVectorizer_ (from __sklearn__) | vect | Saved vectorizer to transform test data |
	'''
	
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
	
'''
if __name__ == "__main__":
	# Bag of word extraction as default
	getBoWf( data, unpickle=_unpkl, verbose=_verb, default=_default )
'''