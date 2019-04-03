
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
from sklearn.decomposition import PCA 
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 
stop_words = set(stopwords.words('english')) 
  
with open ("Dataset.txt", "r") as fp:
    txt=fp.readlines()	

for line in txt:
	tokenized = sent_tokenize(txt) 
	for i in tokenized: 
      
    	# Word tokenizers is used to find the words  
    	# and punctuation in a string 
    	wordsList = nltk.word_tokenize(i) 
  
    	# removing stop words from wordList 
    	wordsList = [w for w in wordsList if not w in stop_words]  
  
    	#  Using a Tagger. Which is part-of-speech  
    	# tagger or POS-tagger.  
    	tagged = nltk.pos_tag(wordsList) 
  
    	print(tagged) 
