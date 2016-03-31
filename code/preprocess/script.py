# coding=latin-1
#
# Last update : 31/03/2016
# Author : Naascraft
# Description : Kaggle tutorial on NLP with Word2Vec [PREPROCESS]

### Import module ###
from bs4 import BeautifulSoup
import pandas as pd
import sys
import re
import copy as cp
import nltk #run nltk.download() locally to install the stopwords corpus and punkt tokenizers
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from time import time
import os

### Command Line Arguments ###
_verb = "-v" in sys.argv

### Path variables ###
dataPath_ = "../../source/data/"
fileDir_ = os.path.dirname(os.path.realpath('__file__'))
picklePath_ = os.path.join( fileDir_, "../pickles/" )

def reSub( text, lSubs ): # Performs a "lSubs" list of regular expression substitutions in the "text" string parameter
	result = cp.copy(text)
	
	for i in range(len(lSubs)):
		result = re.sub(lSubs[i][0], lSubs[i][1], result)
		
	return result

def reTreatment( text, level=0 ):
	# Regex substitution pairs
	noAlphab_reS = ["[^a-zA-Z]", " "]
	num_reS = ["[0-9]+", " NUM "]
	punct_reS = ["[\?!:;\.\"]", " PUNCT "] # We keep each mark as a single token, to emphasize structures like "!!!!" or "!?!?"
	emoticon_reS = ["[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?", " EMO "]
	# 	Regex found on (http://sentiment.christopherpotts.net/tokenizing.html)
	# 	Capture 96% of the emoticon tokens occuring on Twitter
	total_reS = [ emoticon_reS, punct_reS, num_reS, noAlphab_reS ]
	
	return reSub( text, total_reS[-level:] )

stops_ = set( stopwords.words("english") )
def rmStopword( words ): # Removes stopwords from the given llist of tokens
	return [w for w in words if not w in stops_]

porter_stemmer_ = PorterStemmer()
def pStem( words ): # Applies the Porter Stemming algorithm to the given list of tokens
	return [porter_stemmer_.stem(w) for w in words]

def fullPPtoW( review, re_level, sw_drop, stem, join_res=True ):
	result = BeautifulSoup( review )
	result = result.get_text().lower()
	result = reTreatment( result, re_level )
	if sw_drop:
		result = rmStopword( result.split() )
		if stem:
			result = pStem( result )
	else:
		if stem:
			result = pStem( result.split() )
	if join_res:
		return ( " ".join(result) )
	else:
		return result
	
tokenizer_ = nltk.data.load('tokenizers/punkt/english.pickle')
def fullPPtoS( review, re_level, sw_drop, stem, tk=tokenizer_ ):
	sentences = tk.tokenize(review.decode('utf-8').strip())
	result = []
	for s in sentences:
		if len(s) > 0:
			result += [fullPPtoW( s, re_level, sw_drop, stem, join_res=False ).split()]
	
	return result
	
def run( data, verbose=False, re_level=0, sw_drop=True, stem=False, asS=False ):
	print( "\nRunning the data pre-processing with following parameters : \n" + \
		"	Level of precision maintained after the regex simplification : " + str(re_level) + "\n" + \
		"	Dropping of stop words : " + str(sw_drop) + "\n" + \
		"	Porter Stemming algorithm : " + str(stem) + "\n" + \
		"	Result stored as list of sentences for Word2Vec : " + str(asS) + "\n" + \
		"\n" + \
		"	_______ Please wait ... _______ \n \n" )
	t = time()
	pp_data = []
	size = data["review"].size
	
	for i in xrange( 0, size ): # Performs the full pre-processing of the given data accordingly to the parameters
		if i == 1:
			print(pp_data)
		if asS:
			pp_data += fullPPtoS( data["review"][i], re_level, sw_drop, stem )
		
		else:
			pp_data.append( fullPPtoW( data["review"][i], re_level, sw_drop, stem ) )
	
	t = time()-t
	print( "Preprocessing completed. Total time : " + str(t) + " seconds. \n \n")
	
	if verbose:
		# BeautifulSoup preprocessing cleans off any HTML tag
		expl11 = BeautifulSoup( data["review"][0] )
		
		# Converting to lower case any case-based character
		expl12 = expl11.get_text().lower()
		print ( "Example review (text after BeautifulSoup preprocess, lowercase) : \n" + str(expl12) + "\n" )
		
		# Regular expressions allow us to control the granularity we want to capture
		expl21 = reTreatment( expl12, level=0 )
		print ( "Example review (Regex treatment - only alphabetical characters) : \n" + str(expl21) + "\n" )
		
		expl22 = reTreatment( expl12, level=1 )
		print ( "Example review (Regex treatment - numerical characters replaced by 'NUM') : \n" + str(expl22) + "\n" )
		
		expl23 = reTreatment( expl12, level=3 )
		print ( "Example review (Regex treatment - capturing emoticons, numerical characters and simple punctuation) : \n" + str( expl23 ) + "\n" )
		
		# Tokenization and removing "stopwords" with nltk
		expl41, expl42, expl43 = rmStopword(expl21.split()), rmStopword(expl22.split()), rmStopword(expl23.split())
		print( "First 20 tokens after removing stopwords (only alphabetical) : \n" + str( expl41[:20] ) + "\n" )
		
		# [Extension] Stemming with nltk, using the Porter stemming algorithm
		expl51, expl52, expl53 = pStem(expl41), pStem(expl42), pStem(expl43)
		print( "First 20 tokens after removing stopwords (only alphabetical) and Porter stemming : \n" + str( expl51[:20] ) + "\n" )
	
	return pp_data
	
if __name__ == "__main__":
	# Data reading
	train = pd.read_csv( dataPath+"labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
	
	run( train, verbose=_verb )