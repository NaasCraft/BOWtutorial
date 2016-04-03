# coding=latin-1
#
# Last update : 02/04/2016
# Author : Naascraft
# Description : Kaggle tutorial on NLP with Word2Vec [WORD2VEC]

### Import module ###
import pickle as pkl
from time import time
import numpy as np
import pandas as pd
import sys
import os
from gensim.models import word2vec
from sklearn.cluster import KMeans
import main

### Command Line Arguments ###
_verb = "-v" in sys.argv
_unpkl = "-u" in sys.argv
_save= "-s" in sys.argv
_default = "-d" in sys.argv

### Path variables ###
dataPath_, picklePath_, modelPath_ = main.dataPath_, main.picklePath_, main.modelPath_

### Debugging function ###
debug = main.debug

def makeFeatureVec( words, model, num_features ):
	''' git description
+ __makeFeatureVec__( words, model, num_features ) :
    + _does_ : Computes averaging the word vectors obtained by "model" on "words"
    + _returns_ : Features vector (as _ndarray_)
    + _called by_ : __loopFV__
    + _calls_ : __numpy.zeros__, __numpy.add__, __numpy.divide__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | words | List of words to extract features from |
| _W2VModel_ (from __gensim__) | model | Trained word vector representation model |
| _int_ | num_features | Size of word vector representations |
	'''
	
	num_features = model.syn0.shape[1]
	
	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros( (num_features,), dtype="float32" )
	
	nwords = 0.
	index2word_set = set(model.index2word)
	
	for word in words:
		if word in index2word_set: 
			nwords = nwords + 1.
			featureVec = np.add( featureVec,model[word] )
		
	featureVec = np.divide( featureVec, nwords )
	return featureVec

def showClusters ( map, n ):
	''' git description
+ __showClusters__ ( map, n ) :
    + _does_ : Prints the "n" first clusters in "map"
    + _returns_ : _Nothing_
    + _called by_ : ...
        + _[toEdit] should be called in loopFV_
    + _calls_ : _Nothing_
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _dict_ | map | Map of word cluster centroids |
| _int_ | n | Number of first clusters to show |
	'''
	
	# For the first n clusters
	for cluster in xrange(0,n):
		# Print the cluster number  
		print( "\nCluster" + str(cluster) )
		
		# Find all of the words for that cluster number, and print them out
		words = []
		for i in xrange(0,len(map.values())):
			if( map.values()[i] == cluster ):
				words.append(map.keys()[i])
				
		print words

def kMeansFit ( data, num_clusters ):
	''' git description
+ __kMeansFit__ ( data, num_clusters ) :
    + _does_ : Extracts "num_clusters" clusters with KMeans clustering on "data"
    + _returns_ : Centroids indexes (as _list_)
    + _called by_ : __loopFV__
    + _calls_ : __time.time__, __sklearn.cluster.KMeans__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _ndarray_ (from __numpy__) | data | Vocabulary learned by the W2V model |
| _int_ | num_clusters | Number of clusters to compute by KMeans() |
	'''
	
	t = time()
	# Initalize a k-means object and use it to extract centroids
	kmeans_clustering = KMeans( n_clusters = num_clusters )
	idx = kmeans_clustering.fit_predict( data )

	print ("Time taken for K Means clustering: ", time() - t, "seconds.")
	
	return idx
		
def createBagOfCentroids( wordlist, word_centroid_map ):
	''' git description
+ __createBagOfCentroids__( wordlist, word_centroid_map ) :
    + _does_ : Extract bag of centroids (given by "word_centroid_map") features for "wordlist"
    + _returns_ : Features vector (as _ndarray_)
    + _called by_ : __loopFV__
    + _calls_ : __numpy.zeros__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | wordlist | List of words to extract features from |
| _dict_ | word_centroid_map | Map of word cluster centroids |
	'''
	
	num_centroids = max( word_centroid_map.values() ) + 1
	
	# Pre-allocate the bag of centroids vector (for speed)
	bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
	
	# Loop over the words in the review. If the word is in the vocabulary, find which cluster it belongs to, and increment that cluster count by one
	for word in wordlist:
		if word in word_centroid_map:
			index = word_centroid_map[word]
			bag_of_centroids[index] += 1
			
	# Return the "bag of centroids"
	return bag_of_centroids

def loopFV( reviews, model, mode="avg", dump=False ):
	''' git description
+ __loopFV__( reviews, model, mode="avg", dump=False ) :
    + _does_ : Computes a feature vector, according to "mode", for "reviews" given "model" (wich is either saved or loaded depending on "dump")
    + _returns_ : Features vectors set for given data (as _ndarray_)
    + _called by_ : `python main.py -skl`, __submission.run__
    + _calls_ : __kMeansFit__, __pickle.dump__, __numpy.zeros__, __makeFeatureVec__, __createBagOfCentroids__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _list_ of _string_ | reviews | List of reviews as lists of words |
| _W2VModel_ (from __gensim__) | model | Trained word vector representation model |
| _string_ | mode | Feature extraction mode ("avg" or "cluster") |
| _boolean_ | dump | Computes clusters and pickles the word cluster centroids map (else loads it) |
	'''
	
	num_features = model.syn0.shape[1]
	
	# Initialize a counter
	counter = 0.
	
	if mode=="cluster":
		if dump:
			# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
			# average of 5 words per cluster
			word_vectors = model.syn0
			num_clusters = word_vectors.shape[0] / 5

			idx = kMeansFit( word_vectors, num_clusters )
			
			# Create a Word / Index dictionary, mapping each vocabulary word to
			# a cluster number                                                                                            
			word_centroid_map = dict(zip( model.index2word, idx ))
			
			pkl.dump(word_centroid_map, open("pickles/tmp/centroid_map.pkl","wb" ))
		else:
			word_centroid_map = pkl.load(open( "pickles/tmp/centroid_map.pkl","rb" ))
			
		num_features = max( word_centroid_map.values() ) + 1
		
	# Preallocate a 2D numpy array, for speed
	reviewFeatureVecs = np.zeros( (len(reviews),num_features), dtype="float32" )
	
	for review in reviews:
		# Print a status message every 1000th review
		if counter%1000. == 0.:
			print( "Review " + str(counter) + " of " + str( len(reviews) ) )
		
		if mode=="avg":
			reviewFeatureVecs[counter] = makeFeatureVec( review, model, num_features )
		elif mode=="cluster":
			reviewFeatureVecs[counter] = createBagOfCentroids( review, word_centroid_map )
		
		# Increment the counter
		counter = counter + 1.
			
	return reviewFeatureVecs

def load( model_name, rd=True ):
	''' git description
+ __load__( model_name, rd=True ) :
    + _does_ : Loads "model_name" word2vec trained model
    + _returns_ : Loaded model (as _Word2Vec_)
    + _called by_ : __run__
    + _calls_ : __gensim.models.word2vec.Word2Vec.load__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | model_name | Name of model to load |
| _boolean_ | rd | Trims unneeded model memory but prevents model to be trained again |
	'''
	
	result = word2vec.Word2Vec.load(modelPath_+model_name)
	if rd:
		result.init_sims(replace=True)
	
	return result

def notMatch( model, words, verbose ):
	''' git description
+ __notMatch__( model, words, verbose ) :
    + _does_ : Tests (and may print) "model" on ability to fin most dissimilar word amongst "words"
    + _returns_ : Predicted most dissimilar word (as _string_)
    + _called by_ : __modelTesting__
    + _calls_ : _Nothing_
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _W2VModel_ (from __gensim__) | model | Trained word vector representation model |
| _list_ of _string_ | words | List of words to test |
| _boolean_ | verbose | Controls console outputs |
	'''
	
	result = model.doesnt_match( words )
	if verbose: print("Most dissimilar word from " + str(words) + " : " + result )
	
	return result
	
def modelTesting( model ):
	''' git description
+ __modelTesting__( model ) :
    + _does_ : Executes some tests on given "model"
    + _returns_ : _Nothing_
    + _called by_ : __run__
    + _calls_ : __notMatch__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _W2VModel_ (from __gensim__) | model | Trained word vector representation model |
	'''
	
	print("\nSome tests for the model :\n")
	list1 = "man woman child kitchen".split()
	list2 = "france england germany berlin".split()
	list3 = "shirt pant sock flower".split()
	
	notMatch( model=model, words=list1, verbose=True )
	notMatch( model=model, words=list2, verbose=True )
	notMatch( model=model, words=list3, verbose=True )
	
	print( "\nWords similar to vector operation (king - man + woman)\n"+\
		str(model.most_similar(positive=['woman', 'king'], negative=['man'])))

def run( sentences, save=False, default=False, verbose=False, ready=True, lf=False, loadname="" ):
	''' git description
+ __run__( sentences, save=False, default=False, verbose=False, ready=True, lf=False, loadname="" ) :
    + _does_ : Performs a Word2Vec model training with given "sentences" (or loads one)
    + _returns_ : Trained model (as _Word2Vec_), word vectors size (as _int_)
    + _called by_ : `python main.py -skl`
    + _calls_ : __logging.basicConfig__, __logging.INFO__, __load__, __gensim.models.word2vec.Word2Vec__, __modelTesting__
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
	'''
	
	if verbose:
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		
		print( "You required Word2Vec model training.")
	
	if lf:
		model = load( loadname, rd=ready )
		num_features = model.syn0.shape[1]
		
	else:
		if not default:
			print( "Enter the following parameters :" )
			num_features = int( raw_input( "Word vector dimensionality : " ) )
			min_word_count = int( raw_input( "Minimum word count : " ) )
			num_workers = 8 #int( raw_input( "Number of threads to run in parallel : " ) )
			context = 10 #int( raw_input( "Context window size : " ) )
			downsampling = 1e-3 #float( raw_input( "Downsample setting for frequent words : " ) )
		else:
			num_features, min_word_count, num_workers, context, downsampling = 300, 40, 4, 10, 1e-3
		
		if verbose: print( "\nTraining model..." )
		model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, \
						min_count = min_word_count, window = context, sample = downsampling)

		# If you don't plan to train the model any further, calling 
		# init_sims will make the model much more memory-efficient.
		if ready:
			model.init_sims(replace=True)

		# Model saving
		if save:
			model_name = str(num_features)+"features_"+str(min_word_count)+"minwords"
			model.save(modelPath_+model_name)
	
	if verbose:
		modelTesting(model)
	
	return model, num_features
	
'''
if __name__ == "__main__":
	# Word2Vec model build
	run( sentences, save=_save, default=_default, verbose=_verb, ready=True, load=False, loadname="" )
'''