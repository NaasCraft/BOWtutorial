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
	# For the first 10 clusters
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
	t = time()
	# Initalize a k-means object and use it to extract centroids
	kmeans_clustering = KMeans( n_clusters = num_clusters )
	idx = kmeans_clustering.fit_predict( data )

	print ("Time taken for K Means clustering: ", time() - t, "seconds.")
	
	return idx
		
def createBagOfCentroids( wordlist, word_centroid_map ):
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
	# Given a set of reviews (each one a list of words), calculate 
	# the average feature vector for each one and return a 2D numpy array 
	
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
		debug(num_features, "num of centroids")
		
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
	result = word2vec.Word2Vec.load(modelPath_+model_name)
	if rd:
		result.init_sims(replace=True)
	
	return result

def notMatch( model, words, verbose ):
	result = model.doesnt_match( words )
	if verbose: print("Most dissimilar word from " + str(words) + " : " + result )
	
	return result
	
def modelTesting( model ):
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
	
if __name__ == "__main__":
	# Word2Vec model build
	run( sentences, save=_save, default=_default, verbose=_verb, ready=True, load=False, loadname="" )