'''
Building Model
'''

import os
import numpy as np
import pickle

from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model 

DICTI_DUMP_PATH = os.path.join('data','processed_data.pickle')
# loading data
with open(DICTI_DUMP_PATH, 'rb') as f:
	dataDicti = pickle.load(f)

# hyper-parameters
MAX_LEN_ENG = dataDicti['maxLenEng']
MAX_LEN_FR = dataDicti['maxLenFr']
ENCODER_TIME_STEPS = MAX_LEN_ENG
DECODER_TIME_STEPS = MAX_LEN_FR
ENCODER_VOCAB = dataDicti['vocabEng']
DECODER_VOCAB = dataDicti['vocabFr']
ENCODER_VOCAB_SIZE = len(dataDicti['vocabEng'])
DECODER_VOCAB_SIZE= len(dataDicti['vocabFr'])
DATA_ENG = dataDicti['Eng']
DATA_FR = dataDicti['Fr']
DATA_SIZE = len(DATA_ENG)
EMBED_DIMS = 50
BATCH_SIZE = 64
ENCODER_HIDDEN_UNITS = 30
DECODER_HIDDEN_UNITS = 30
EPOCHS = 100
SEED = 2

def custom_placeholders():
	# input variables
	#encoder input -> (dataSize, encoderTimeSteps) 

	# we will give all data and let keras generate batches
	# here embedDims hasn't been added as 3rd dims because I will use keras to generate corresponding
	# embedding
	#decoder input -> (dataSize, decoderTimeSteps, decoderVocabSize) 
	# as this is going to be softmax probabilty dist.
	# decoder_target -> (dataSize, decoderTimeSteps decoderVocabSize)

	# initializing variables
	encoderInputsPlace = np.zeros((DATA_SIZE, ENCODER_TIME_STEPS), dtype=np.float64)
	decoderInputsPlace = np.zeros((DATA_SIZE, DECODER_TIME_STEPS, DECODER_VOCAB_SIZE), dtype=np.float64)
	decoderTargetsPlace = np.zeros((DATA_SIZE, DECODER_TIME_STEPS, DECODER_VOCAB_SIZE), dtype=np.float64)

	return encoderInputsPlace, decoderInputsPlace, decoderTargetsPlace
'''
Here, we need to understand that these input variables have been created with respect
to training the encoder-decoder model. Hence irrespective of what decoder produes with
initializing from encoder state, we have to give decoder ideal input as well as ideal target
for output. During testing, both decoderInputs and decoderTargets will be absent
'''
def filling_data(encoderInputsPlace, decoderInputsPlace, decoderTargetsPlace, engVocab, frVocab, dataEngNew, dataFrNew):
	
	# adding data to input variables
	for i, (engSen, frSen) in enumerate(zip(dataEngNew, dataFrNew)):
		#adding each word of a eng sen as time step into encoder ith input
		for t, word in enumerate(engSen.split()):
			encoderInputsPlace[i][t] = engVocab[word]

		# adding each word of corresponding french sen into decoder ith input
		for t, word in enumerate(frSen.split()):
			decoderInputsPlace[i][t] = frVocab[word]

			# decoder input word at t'th time step is the decoder target output for
			# t-1 th step
			if t>0:
				decoderTargetsPlace[i][t-1][frVocab[word]] = 1
				# here we will add 1 at the index of word which has to be target, 
				# as it is a prob dist of Fr vocab 
	return encoderInputsPlace, decoderInputsPlace, decoderTargetsPlace

def build_model_graph():

	#NOTE:- Here Sequential Model hasn't been used, hence style is different

	# initializing input tensor for encoder
	encoderRawInp = Input(shape = (BATCH_SIZE,))

	#creating encoder embeddings
	encoderEmbed = Embedding(ENCODER_VOCAB_SIZE, EMBED_DIMS)
	# replacing token with embedding
	encoderInp = encoderEmbed(encoderRawInp)

	# creating encoder
	encoder = LSTM(ENCODER_HIDDEN_UNITS, return_state = True)
	# return_state True,want encoder to return state also 

	# encoder outputs and states
	encoderOut, encoderStateH, encoderStateC = encoder(encoderInp)
	encoderStates = [encoderStateH, encoderStateC]




	#initializing input tensor for decoder
	decoderRawInp = Input(shape = (BATCH_SIZE,))

	# creating decoder embeddings
	decoderEmbed = Embedding(DECODER_VOCAB_SIZE, EMBED_DIMS)
	decoderInp = decoderEmbed(decoderRawInp)

	#creating decoder
	decoder = LSTM(DECODER_HIDDEN_UNITS, return_state= True, return_sequences = True)
	# here ret_seq is True as we want out of decoder at each time step
	decoderOut, _, _ = decoder(decoderInp, initial_state = encoderStates)
	# initialize decoder with encoder state



	# passing decoder output through fully connected layers

	denseLayer = Dense(DECODER_VOCAB_SIZE, activation = 'softmax')
	# softmax to calculate word probability and take out mosr-probable word

	decoderOutDense = denseLayer(decoderOut)



	model = Model([encoderRawInp, decoderRawInp], decoderOut)

	# summary
	model.summary()

	return model

def train(myModel, encoderInputsData, decoderInputsData, decoderTargetsData):

	myModel.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['acc'] )

	# training
	myMmodel.fit([encoderInputsData, decoderInputsData], decoderTargetsData, batch_size=BATCH_SIZE,
				epochs = EPOCHS, validation_split = 0.1)

myModel = build_model_graph()

encoderInputsPlace, decoderInputsPlace, decoderTargetsPlace = custom_placeholders()

encoderInputsData, decoderInputsData, decoderTargetsData = filling_data(encoderInputsPlace, decoderInputsPlace, decoderTargetsPlace,
																		ENCODER_VOCAB,DECODER_VOCAB, DATA_ENG, DATA_FR)

train(myModel, encoderInputsData, decoderInputsData, decoderTargetsData)





