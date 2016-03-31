# coding=latin-1
#
# Last update : 31/03/2016
# Author : Naascraft
# Description : Kaggle tutorial on NLP with Word2Vec

### Import module ###
import pandas as pd
import sys
import pickle as pkl
import os
from time import time

### Command Line Arguments ###
_verb = "-v" in sys.argv
_segments = ["-pp", "-skl", "-rf", "-s"]
_toRun = [ _segments[k] for k in range(4) if _segments[k] in sys.argv ]
print(sys.argv, _segments, _toRun)
_pickleData = "-p" in sys.argv
_help = "-h" in sys.argv

### Path variables ###
dataPath_ = "../source/data/"
fileDir_ = os.path.dirname(os.path.realpath('__file__'))
picklePath_ = os.path.join( fileDir_, "pickles/" )

if __name__ == "__main__":
	if _help:
		print( "Help message yet to be written... TODO" )
	else:
		# Data reading
		train = pd.read_csv( dataPath_+"labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
	
		if _verb:
			print( "Shape of labeled training dataset : \n" + str( train.shape ) + "\n" )
			#print( "Example review (raw) : \n" + str( train["review"][0] ) + "\n" )
	
		# Data processing
		if "-pp" in _toRun:
			print( "You required data pre processing. Enter the following parameters : \n" )
			in_re_level = int( raw_input( "Regex process level (0-3) : " ) )
			in_sw_drop = raw_input( "Do you want to keep stop words ? (Y/N) : " ) == "N"
			in_stem = raw_input( "Do you want to apply Porter Stemming ? (Y/N) : ") == "Y"
			
			import preprocess.script
			ppTrain = preprocess.script.run( train, verbose=_verb, re_level=in_re_level, sw_drop=in_sw_drop, stem=in_stem )
			
			### Pickling of pre-processed data ###
			if _pickleData:
				pkl.dump( ppTrain, open( "pickles/ppTrainData.pkl","wb" ) )
				if _verb: 
					print("Pickled pre-processed data into 'ppTrainData.pkl' file." + \
						"(Size : " + str( os.path.getsize("pickles/ppTrainData.pkl") / 1000.00 ) + " Kilobytes. \n\n")
		
		if "-skl" in _toRun:
			import sklmodels.script
			
			if "-pp" not in _toRun:
				t = time()
				ppTrain = pkl.load( open( picklePath_+"ppTrainData.pkl", "rb" ) )
				if _verb: print("\nPreprocessed data unpickling successfully completed in " + str( time()-t ) + " seconds.")
				
			bowFeatures = sklmodels.script.getBoWf( data=ppTrain, verbose=_verb )
			print("debug1")
			
			''' Do not try to pickle this, with 200 features it already reaches 150MB !
			### Pickling of bag of words extracted features ###
			if _pickleData:
				pkl.dump( bowFeatures, open( "pickles/bowFeatures.pkl","wb" ) )
				if _verb: 
					print("Pickled bag of words extracted features into 'bowFeatures.pkl' file." + \
						"(Size : " + str( os.path.getsize("pickles/bowFeatures.pkl") / 1000.00 ) + " Kilobytes. \n\n")
			'''
			
			if "-rf" in _toRun:
				in_n_est = int( raw_input( "Enter number of estimators for the Random Forest classifier : " ) )
				rf = sklmodels.script.buildRF( features=bowFeatures, label=train["sentiment"], n_est=in_n_est, verbose=_verb)
				
				''' Same size issues
				### Pickling of trained random forest classifier ###
				if _pickleData:
					pkl.dump( rf, open( "pickles/bowRF.pkl","wb" ) )
					if _verb: 
						print("Pickled trained random forest classifier into 'bowRF.pkl' file." + \
							"(Size : " + str( os.path.getsize("pickles/bowRF.pkl") / 1000.00 ) + " Kilobytes. \n\n")
				'''
			
		if "-s" in _toRun:
			import submission.script
			
			rfPred = submission.script.run( model=rf, modelID="RF", verb=True, re_level_=1, sw_drop_=True, stem_=True )