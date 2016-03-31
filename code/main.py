# coding=latin-1
#
# Last update : 31/03/2016
# Author : Naascraft
# Description : Kaggle tutorial on NLP with Word2Vec [MAIN]

### Import module ###
import pandas as pd
import sys
import pickle as pkl
import os
from time import time

### Command Line Arguments ###
_verb = "-v" in sys.argv
#_segments = ["-pp", "-skl", "-rf", "-s"]
#_toRun = [ _segments[k] for k in range(len(segments_)) if _segments[k] in sys.argv ]
#print(sys.argv, _segments, _toRun)
_pickleData = "-p" in sys.argv
_help = "-h" in sys.argv
_default = "-d" in sys.argv
_ppRun = "-pp" in sys.argv
_sklRun = "-skl" in sys.argv
_rfRun = "-rf" in sys.argv
_sRun = "-s" in sys.argv
_noData = "-nd" in sys.argv

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
		in_asS = raw_input( "Do you want to use Word2Vec ? (Y/N) : ") == "Y"
		
		if _verb:
			print( "Shape of labeled training dataset : \n" + str( train.shape ) + "\n" )
			#print( "Example review (raw) : \n" + str( train["review"][0] ) + "\n" )
	
		# Data processing
		if "-pp" in sys.argv:
			print( "You required data pre processing. Enter the following parameters : \n" )
			in_re_level = int( raw_input( "Regex process level (0-3) : " ) )
			in_sw_drop = raw_input( "Do you want to keep stop words ? (Y/N) : " ) == "N"
			in_stem = raw_input( "Do you want to apply Porter Stemming ? (Y/N) : ") == "Y"
			
			import preprocess.script
			
			ppfilename = "ppTrainData"
			
			if in_asS:
				ppfilename += "_sentences"
				ul_train = pd.read_csv( dataPath_+"unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
				
				ul_ppTrain = preprocess.script.run( ul_train, verbose=_verb, re_level=in_re_level, sw_drop=in_sw_drop, stem=in_stem, asS=in_asS )
			
				### Pickling of pre-processed data ###
				if _pickleData:
					pkl.dump( ul_ppTrain, open( "pickles/ul_"+str(ppfilename)+".pkl","wb" ) )
					if _verb: 
						print("Pickled pre-processed unlabeled data into 'ul_"+str(ppfilename)+".pkl' file." + \
							"(Size : " + str( os.path.getsize("pickles/ul_"+str(ppfilename)+".pkl") / 1000.00 ) + " Kilobytes. \n\n")
			
			ppTrain = preprocess.script.run( train, verbose=_verb, re_level=in_re_level, sw_drop=in_sw_drop, stem=in_stem, asS=in_asS )
		
			### Pickling of pre-processed data ###
			if _pickleData:
				pkl.dump( ppTrain, open( "pickles/"+str(ppfilename)+".pkl","wb" ) )
				if _verb: 
					print("Pickled pre-processed labeled data into '"+str(ppfilename)+".pkl' file." + \
						"(Size : " + str( os.path.getsize("pickles/"+str(ppfilename)+".pkl") / 1000.00 ) + " Kilobytes. \n\n")
							
		if "-skl" in sys.argv:
			import sklmodels.script
			
			if (not _noData) and ("-pp" not in sys.argv):
				ppfilename = "ppTrainData"
				if in_asS:
					t = time()
					ppfilename += "_sentences"
					ul_ppTrain = pkl.load( open( picklePath_+"ul_"+str(ppfilename)+".pkl", "rb" ) )
				else:
					t=time()
				
				ppTrain = pkl.load( open( picklePath_+str(ppfilename)+".pkl", "rb" ) )
				print(len(ppTrain), ppTrain[:10])
				if in_asS: ppTrain += ul_ppTrain
				if _verb: print("\nPreprocessed data unpickling successfully completed in " + str( time()-t ) + " seconds.")
			else:
				ppTrain=[]
			
			while in_asS:
				import w2v.script
				if _default:
					w2vModel = w2v.script.run( sentences=ppTrain, default=True, verbose=_verb )
				else:
					print( "Info required for word2vec model training :")
					in_load = raw_input( "Do you want to load an existing file ? (Y/N) : " ) == "Y"
					in_loadname = raw_input( "... File name (if not loading, just hit enter) : " )
					in_ready = raw_input( "Do you want to train this model again later ? (Y/N) : " ) == "N"
					in_save = raw_input( "Do you want to save this model for future loading ? (Y/N) : " )
					w2vModel = w2v.script.run( sentences=ppTrain, save=in_save, default=False, verbose=_verb, ready=in_ready, lf=in_load, loadname=in_loadname )
				
				if raw_input("\nDo you want to train another model ? (Y/N) : ") == "N":
					break
				
			else:
				bowFeatures = sklmodels.script.getBoWf( data=ppTrain, verbose=_verb, default=_default )
			
			''' Do not try to pickle this, with 200 features it already reaches 150MB !
			### Pickling of bag of words extracted features ###
			if _pickleData:
				pkl.dump( bowFeatures, open( "pickles/bowFeatures.pkl","wb" ) )
				if _verb: 
					print("Pickled bag of words extracted features into 'bowFeatures.pkl' file." + \
						"(Size : " + str( os.path.getsize("pickles/bowFeatures.pkl") / 1000.00 ) + " Kilobytes. \n\n")
			'''
			
			if "-rf" in sys.argv:
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
			
		if "-s" in sys.argv:
			import submission.script
			
			rfPred = submission.script.run( model=rf, modelID="RF", verb=True, re_level_=0, sw_drop_=True, stem_=True )