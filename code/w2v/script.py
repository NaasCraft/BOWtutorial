# coding=latin-1
#
# Last update : 31/03/2016
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

### Command Line Arguments ###
_verb = "-v" in sys.argv
_unpkl = "-u" in sys.argv
_save= "-s" in sys.argv
_default = "-d" in sys.argv

### Path variables ###
dataPath_ = "../../source/data/"
fileDir_ = os.path.dirname(os.path.realpath('__file__'))
picklePath_ = os.path.join( fileDir_, "../pickles/" )
modelPath_ = "models/"

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
	
	print( "\nWords similar to vector operation (king - man + woman)\n"+str(model.most_similar(positive=['woman', 'king'], negative=['man'])))

def run( sentences, save=False, default=False, verbose=False, ready=True, lf=False, loadname="" ):
	if verbose:
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		
		print( "You required Word2Vec model training.")
	
	if lf:
		model = load( loadname, rd=ready )
		
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
		model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

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
	
	return model
	
if __name__ == "__main__":
	# Word2Vec model build
	run( sentences, save=_save, default=_default, verbose=_verb, ready=True, load=False, loadname="" )