
'''
Data preparation for Neural Machine Translation from English to French Using Seq2Seq

Data link--> https://github.com/devm2024/nmt_keras/blob/master/fra.txt

Using Wget, I've downloaded data into 'data' directory
'''
import numpy as np
import os
import re
import string
import pickle
from pprint import pprint
import matplotlib.pyplot as plt
from nltk import FreqDist
plt.style.use('ggplot')


DATA_PATH = os.path.abspath('data/fra.txt')
DATA_DUMP_PATH = os.path.abspath('data/processed_data.pickle')

START_TOKEN = '<GO>'
END_TOKEN = '<EOS>'

MAX_LEN_ENG = 15
MAX_LEN_FR = 25


def read_data(path):
	with open(path, 'r') as f:
		rawData = f.readlines()
	if rawData == []:
		print('file empty or couldn\'t read')
		return
	else:
		print('data read with no. of lines: ', len(rawData))
		return rawData

def pre_process(rawData):
	# lower case
	data = [s.lower() for s in rawData]

	#removing punctuations
	translateTable = str.maketrans('', '', string.punctuation)
	data = [s.translate(translateTable) for s in data]

	# striping '\n' and spliting on '\t'
	dataEng = [d.rstrip('\n').split('\t')[0] for d in data]

	# alot of french text has '\u202f' character in the end, also strippping it
	dataFr = [re.sub('\u202f', '', d.rstrip('\n').split('\t')[1]) for d in data]

	# Adding <GO> and <EOS> tokens at the start and end of french sentence
	# as it will go into decoder
	for i in range(len(dataFr)):
		dataFr[i] = ' '.join([START_TOKEN, dataFr[i], END_TOKEN])

	return dataEng, dataFr

def build_vocab(processData):

	vocab = {}
	# adding START_TOKEN AND END_TOKEN at 0 and 1
	vocab[START_TOKEN] = 0
	vocab[END_TOKEN] = 1

	index = 2
	for sen in processData:
		for word in sen.split():
			if word not in vocab:
				vocab[word] = index
				index += 1

	print('Length of vocab: ', len(vocab))
	return vocab

def inverse_vocab(vocab):
	return dict([ (v, k) for k, v in vocab.items() ])

'''
Here we are not making the length of input to encoder and input to decoder variable length
(i.e, dynamic)
Hence, we will make a plot of length of sentence V/s frequency
(number of occurrence of that length) so as to choose the optimal length
'''

def senlen_vs_freq_plot(data, title = '', save_name = 'len_freq_plot.png'):

	seqLenList = [len(d.split()) for d in data]
	dist = dict(FreqDist(seqLenList))

	plt.close()
	plt.bar(dist.keys(), dist.values())
	plt.xticks(range(1, max(seqLenList), 5) )
	plt.title(title)
	plt.savefig(save_name)

	plt.show()

'''
Hence we see that most of the sentences are having length around 10, and they are negligible 
after length 15, 
Hence we will sample only those data where the length of english sentence is less than 15
 and french is less than 25
'''

def filter_data(dataEng, dataFr):
	dataEngNew = []
	dataFrNew = []
	assert (len(dataEng) == len(dataFr)), "Length of both list doesn't match"

	for i in range(len(dataEng)):
		if len(dataEng[i].split()) <= MAX_LEN_ENG and len(dataFr[i].split()) <= MAX_LEN_FR:
			dataEngNew.append(dataEng[i])
			dataFrNew.append(dataFr[i])

	print('dropped sentences count: ', len(dataEng) - len(dataEngNew))
	return dataEngNew, dataFrNew

def pickle_dump(dumpData,path):

	with open(path, 'wb') as f:
		pickle.dump(dumpData, f)
	print('saved processed data at: ', path)

rawData = read_data(DATA_PATH)
dataEng, dataFr = pre_process(rawData)

wordIndexEngVocab = build_vocab(dataEng)
wordIndexFrVocab = build_vocab(dataFr)

indexWordEngVocab = inverse_vocab(wordIndexEngVocab)
indexWordFrVocab = inverse_vocab(wordIndexFrVocab)

senlen_vs_freq_plot(dataEng, title = 'Plot for English', save_name = 'english.png')
senlen_vs_freq_plot(dataFr, title = 'Plot for French', save_name = 'french.png')

dataEngNew, dataFrNew = filter_data(dataEng, dataFr)

dumpDicti = {'Eng': dataEngNew,'Fr': dataFrNew,'vocabEng': wordIndexEngVocab,
			'vocabFr': wordIndexFrVocab, 'inverseVocabEng': indexWordEngVocab,
			'inverseVocabFr': indexWordFrVocab, 'maxLenEng' : MAX_LEN_ENG, 
			'maxLenFr' : MAX_LEN_FR}
pickle_dump(dumpDicti, DATA_DUMP_PATH)



