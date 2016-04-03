# coding=latin-1
#
# Last update : 02/04/2016
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
import main

### Command Line Arguments ###
_verb = "-v" in sys.argv
_default = "-d" in sys.argv

### Path variables ###
dataPath_, picklePath_ = main.dataPath_, main.picklePath_

### Debugging function ###
debug = main.debug

def reSub( text, lSubs ):
	''' git description
+ __reSub__( text, lSubs ) :
    + _does_ : Performs a "lSubs" list of regular expression substitutions in the "text" string parameter
    + _returns_ : Treated text (as _string_)
    + _called by_ : __reTreatment__
    + _calls_ : __copy.copy__, __re.sub__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | text | Text to be treated |
| _list_ of _list_ of _string_ | lSubs | List of Regex substitution pairs |
	'''
	
	result = cp.copy(text)
	
	for i in range(len(lSubs)):
		result = re.sub(lSubs[i][0], lSubs[i][1], result)
		
	return result

def reTreatment( text, level=0 ):
	''' git description
+ __reTreatment__( text, level=0 ) :
    + _does_ : Performs a Regex treatment on "text" string as defined by "level" parameter (0 keeps _only alphabetica_l chars, 1 adds _numbers_ as "num", 2 adds _punctuation_ as "punct", 3 adds _emoticons_ as "emo")
    + _returns_ : Treated text (as _string_)
    + _called by_ : __fullPPtoW__
    + _calls_ : __reSub__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | text | Text to be treated |
| _int_ | level | Level of Regex treatment (0-3) |
	'''
	
	# Regex substitution pairs
	noAlphab_reS = ["[^a-zA-Z]+", " "]
	num_reS = ["[0-9]+", " NUM "]
	punct_reS = ["[\?!:;\.\"]", " PUNCT "] # We keep each mark as a single token, to emphasize structures like "!!!!" or "!?!?"
	emoticon_reS = ["[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?", " EMO "]
	# 	Regex found on (http://sentiment.christopherpotts.net/tokenizing.html)
	# 	Capture 96% of the emoticon tokens occuring on Twitter
	total_reS = [ emoticon_reS, punct_reS, num_reS, noAlphab_reS ]
	
	return reSub( text, total_reS[-(level+1):] )

def rmStopword( words ):
	''' git description
+ __rmStopword__( words ) :
    + _does_ : Removes stopwords from the given "words" list of tokens
    + _returns_ : Treated list of words (as _list_)
    + _called by_ : __fullPPtoW__
    + _calls_ : __nltk.corpus.stopwords__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | words | List of tokens to be treated |
	'''
	
	stops_ = set( stopwords.words("english") )
	return [w for w in words if not w in stops_]

def pStem( words ): 
	''' git description
+ __pStem__( words ) : Applies the Porter Stemming algorithm to the given list of tokens
    + _does_ : Applies the Porter Stemming algorithm to the given "words" list of tokens
    + _returns_ : Treated list of words (as _list_)
    + _called by_ : __fullPPtoW__
    + _calls_ : __nltk.stem.porter.PorterStemmer__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _list_ of _string_ | words | List of tokens to be treated |
	'''
	
	porter_stemmer_ = PorterStemmer()
	return [porter_stemmer_.stem(w) for w in words]

def fullPPtoW( review, re_level, sw_drop, stem, join_res=True ):
	''' git description
+ __fullPPtoW__( review, re_level, sw_drop, stem, join_res=True ) :
    + _does_ : Computes a full "review" string pre-processing, according to ("re_level", "sw_drop", "stem") parameters
    + _returns_ : Treated "review" (as _list_ or _string_ depending on "join_res")
    + _called by_ : __fullPPtoS__, __run__, __submission.run__
    + _calls_ : __reTreatment__, __rmStopWords__, __pStem__, __bs4.BeautifulSoup__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | review | Review to be pre-processed |
| _int_ | re_level | Level of Regex treatment (0-3) |
| _boolean_ | sw_drop | Should drop stop words |
| _boolean_ | stem | Should apply Porter Stemming |
| _boolean_ | join_res | Should return result as string (else as list of words) |
	'''
	
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
		else:
			result = result.split()
	if join_res:
		return ( " ".join(result) )
	else:
		return result
	
tokenizer_ = nltk.data.load('tokenizers/punkt/english.pickle')
def fullPPtoS( review, re_level, sw_drop, stem, tk=tokenizer_ ):
	''' git description
+ __fullPPtoS__( review, re_level, sw_drop, stem, tk=tokenizer_ ) :
    + _does_ : Computes a full "review" string pre-processing into sentences split by "tk", according to ("re_level", "sw_drop", "stem") parameters
    + _returns_ : Treated review into list of sentences (as _list_)
    + _called by_ : __run__
    + _calls_ : __fullPPtoW__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _string_ | review | Review to be pre-processed into sentences |
| _int_ | re_level | Level of Regex treatment (0-3) |
| _boolean_ | sw_drop | Should drop stop words |
| _boolean_ | stem | Should apply Porter Stemming |
| _Tokenizer_ (from __nltk__) | tk | Tokenizer to split into sentences |
	'''
	
	sentences = tk.tokenize(review.decode('utf-8').strip())
	result = []
	for s in sentences:
		if len(s) > 0:
			result += [fullPPtoW( s, re_level, sw_drop, stem, join_res=False )]
	
	return result
	
def run( data, verbose=False, re_level=0, sw_drop=True, stem=False, asW2V=False ):
	''' git description
+ __run__( data, verbose=False, re_level=0, sw_drop=True, stem=False, asW2V=False ) :
    + _does_ : Full "data" pre-processing according to given parameters
    + _returns_ : (__2__ values)
        + if "asW2V" : Treated data as sentences (as _list_), ... as words (as _list_)
        + else : Treated data as words (as _list_), empty list
    + _called by_ : `python main.py -pp`, __submission.run__
    + _calls_ : __time.time__, __fullPPtoS__, __fullPPtoW__
    + _arguments_ :
        
| type | name | description |
| --- | --- | --- |
| _ndarray_ (from __numpy__) | data | Dataset to be pre-processed |
| _boolean_ | verbose | Controls console outputs |
| _int_ | re_level | Level of Regex treatment (0-3) |
| _boolean_ | sw_drop | Should drop stop words |
| _boolean_ | stem | Should apply Porter Stemming |
| _boolean_ | asW2V | Should use sentences (else just words) |
	'''
	
	print( "\nRunning the data pre-processing with following parameters : \n" + \
		"	Level of precision maintained after the regex simplification : " + str(re_level) + "\n" + \
		"	Dropping of stop words : " + str(sw_drop) + "\n" + \
		"	Porter Stemming algorithm : " + str(stem) + "\n" + \
		"	Result stored as list of sentences for Word2Vec : " + str(asW2V) + "\n" + \
		"\n" + \
		"	_______ Please wait ... _______ \n \n" )
	t = time()
	pp_data, pp_data_w = [], []
	size = data["review"].size
	
	for i in xrange( 0, size ): # Performs the full pre-processing of the given data accordingly to the parameters
		if asW2V:
			pp_data += fullPPtoS( data["review"][i], re_level, sw_drop, stem )
		
			pp_data_w.append( fullPPtoW( data["review"][i], re_level, sw_drop, stem, join_res=False ) )
		else:
			pp_data_w.append( fullPPtoW( data["review"][i], re_level, sw_drop, stem, join_res=True ) )
	
	t = time()-t
	print( "Preprocessing completed. Total time : " + str(t) + " seconds.\n\n" )
	
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
	
	if pp_data:
		return pp_data, pp_data_w
	else:
		return pp_data_w, []
'''	
if __name__ == "__main__":
	# Data reading
	train = pd.read_csv( dataPath+"labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
	
	run( train, verbose=_verb )
'''