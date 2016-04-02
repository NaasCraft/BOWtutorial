# coding=latin-1
#
# Last update : 02/04/2016
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
_pickleData = "-p" in sys.argv
_help = "-h" in sys.argv
_default = "-d" in sys.argv
_noData = "-nd" in sys.argv
_ppRun = "-pp" in sys.argv
_feRun = "-fe" in sys.argv
_mRun = "-m" in sys.argv
_sRun = "-s" in sys.argv
_noData = "-nd" in sys.argv

### Path variables ###
fileDir_ = os.path.dirname(os.path.realpath('__file__'))
dataPath_ = os.path.join( fileDir_, "../source/data/")
picklePath_ = os.path.join( fileDir_, "pickles/" )
outPath_ = os.path.join( fileDir_, "submission/")
modelPath_ = os.path.join( fileDir_, "models/")

def debug( var, repr ):
	print( "debug "+repr+" : "+str(var) )

if __name__ == "__main__":
	if _help:
		print( "Help message yet to be written... TODO" )
	else:
		# Data reading
		train = pd.read_csv( dataPath_+"labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
		in_asW2V = raw_input( "Do you want to use Word2Vec ? (Y/N) : ") == "Y"
		
		if _verb:
			print( "Shape of labeled training dataset : \n" + str( train.shape ) + "\n" )
			#print( "Example review (raw) : \n" + str( train["review"][0] ) + "\n" )
	
		# Data processing
		if "-pp" in sys.argv:
			print( "You required data pre processing. Enter the following parameters : \n" )
			in_re_level = int( raw_input( "Regex process level (0-3) : " ) )
			in_sw_drop = raw_input( "Do you want to keep stop words ? (Y/N) : " ) == "N"
			in_stem = raw_input( "Do you want to apply Porter Stemming ? (Y/N) : ") == "Y"
			
			import preprocess
			
			ppfilename = "ppTrainData"
			
			if in_asW2V:
				ppfilename += "_sentences"
				ul_train = pd.read_csv( dataPath_+"unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
				
				ul_ppTrain, _empt_ = preprocess.run( ul_train, verbose=_verb, re_level=in_re_level, sw_drop=in_sw_drop, stem=in_stem, asW2V=in_asW2V )
			
				### Pickling of pre-processed data ###
				if _pickleData:
					pkl.dump( ul_ppTrain, open( "pickles/ul_"+str(ppfilename)+".pkl","wb" ) )
					if _verb: 
						print("Pickled pre-processed unlabeled data into 'ul_"+str(ppfilename)+".pkl' file." + \
							"(Size : " + str( os.path.getsize("pickles/ul_"+str(ppfilename)+".pkl") / 1000.00 ) + " Kilobytes. \n\n")
			
			ppTrain, ppTrainW = preprocess.run( train, verbose=_verb, re_level=in_re_level, sw_drop=in_sw_drop, stem=in_stem, asW2V=in_asW2V )
			
			### Pickling of pre-processed data ###
			if _pickleData:
				pkl.dump( ppTrain, open( "pickles/"+str(ppfilename)+".pkl","wb" ) )
				pkl.dump( ppTrainW, open( "pickles/"+str(ppfilename)+"W.pkl","wb" ) )
				if _verb: 
					print("Pickled pre-processed labeled data into '"+str(ppfilename)+".pkl' file." + \
						"(Size : " + str( os.path.getsize("pickles/"+str(ppfilename)+".pkl") / 1000.00 ) + " Kilobytes. \n\n")
							
		if "-skl" in sys.argv:
			import sklmodels
			
			if (not _noData) and ("-pp" not in sys.argv):
				ppfilename = "ppTrainData"
				if in_asW2V:
					t = time()
					ppfilename += "_sentences"
					ul_ppTrain = pkl.load( open( picklePath_+"ul_"+str(ppfilename)+".pkl", "rb" ) )
					ppTrainW = pkl.load( open( picklePath_+str(ppfilename)+"W.pkl", "rb" ) )
				else:
					t=time()
				
				ppTrain = pkl.load( open( picklePath_+str(ppfilename)+".pkl", "rb" ) )
				debug(ppTrain[0], "ppTrain[0]")
				
				if in_asW2V: ppTrain += ul_ppTrain
				if _verb: print( "\nPreprocessed data unpickling successfully completed in " + str( time()-t ) + " seconds." )
				
			else:
				ppTrain=[]
			
			while in_asW2V:
				import w2v
				if _default:
					w2vModel = w2v.run( sentences=ppTrain, default=True, verbose=_verb )
				else:
					#print( "Info required for word2vec model training :")
					in_load = True #raw_input( "Do you want to load an existing file ? (Y/N) : " ) == "Y"
					in_loadname = "300features_40minwords" #raw_input( "... File name (if not loading, just hit enter) : " )
					in_ready = True # raw_input( "Do you want to train this model again later ? (Y/N) : " ) == "N"
					in_save = True # raw_input( "Do you want to save this model for future loading ? (Y/N) : " )
					
					w2vModel, n_f = w2v.run( sentences=ppTrain, save=in_save, default=False, verbose=_verb, ready=in_ready, lf=in_load, loadname=in_loadname )
				
				if True: #raw_input("\nDo you want to train another model ? (Y/N) : ") == "N":
					in_mode = raw_input("Choose mode (avg, cluster) : ")
					debug(ppTrainW[0], "ppTrainW[0]")
					if in_mode == "cluster":
						in_dump = raw_input("Load stored clusters map ? (Y/N) : ") == "N"
						trainFeatures = w2v.loopFV( ppTrainW, w2vModel, mode="cluster", dump=in_dump )
					else:
						trainFeatures = w2v.loopFV( ppTrainW, w2vModel, mode="avg" )
					break
				
			if not in_asW2V:
				trainFeatures, m_f, vectorizer = sklmodels.getBoWf( data=ppTrain, verbose=_verb, default=_default )
				in_mode = False
				w2vModel = None
			
			''' Do not try to pickle this, with 200 features it already reaches 150MB !
			### Pickling of bag of words extracted features ###
			if _pickleData:
				pkl.dump( bowFeatures, open( "pickles/bowFeatures.pkl","wb" ) )
				if _verb: 
					print("Pickled bag of words extracted features into 'bowFeatures.pkl' file." + \
						"(Size : " + str( os.path.getsize("pickles/bowFeatures.pkl") / 1000.00 ) + " Kilobytes. \n\n")
			'''
			
			if "-f" in sys.argv:
				toScale = raw_input( "Do you want to automatically scale features before fitting (Y/N) ? : ") == "Y"
				if _verb: print( "Example train feature (before scaling) : \n" + str( trainFeatures[0] ) + "\n" )
				if in_mode=="cluster":
					debug(sum(trainFeatures[0]), "sum(trainFeatures[0])")
				if toScale:
					scaler = sklmodels.dataScaler( trainFeatures )
					trainFeatures = scaler.transform( trainFeatures )
				else:
					scaler = None
					
				in_modelMode = raw_input( "Enter classifier to use (rf, svm, knn) : " )
				modelFit = sklmodels.buildModel( features=trainFeatures, label=train["sentiment"], mode=in_modelMode, verbose=_verb)
				
				''' Same size issues
				### Pickling of trained random forest classifier ###
				if _pickleData:
					pkl.dump( rf, open( "pickles/bowRF.pkl","wb" ) )
					if _verb: 
						print("Pickled trained random forest classifier into 'bowRF.pkl' file." + \
							"(Size : " + str( os.path.getsize("pickles/bowRF.pkl") / 1000.00 ) + " Kilobytes. \n\n")
				'''
			
				if "-s" in sys.argv:
					import submission
					
					if in_asW2V:
						in_modelMode += "_" + str(in_mode)
						m_f=0
						vectorizer=None
						
					in_re_level = int( raw_input( "Regex process level (0-3) : " ) )
					in_sw_drop = raw_input( "Do you want to keep stop words ? (Y/N) : " ) == "N"
					in_stem = raw_input( "Do you want to apply Porter Stemming ? (Y/N) : ") == "Y"
					pred = submission.run( model=modelFit, modelID=in_modelMode, verb=True, re_level=in_re_level, sw_drop=in_sw_drop, stem=in_stem, max_f=m_f, vect=vectorizer, mode=in_mode, wordModel=w2vModel, scale=toScale, dScaler=scaler )