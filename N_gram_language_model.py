# Import required libraries

from nltk import ngrams
from nltk import FreqDist
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import pad_both_ends
from itertools import product
import math
import numpy as np

def load_data(data_dir):
  
  # data_dir is the directory for the data

  train_data_path = data_dir + "/train.txt"
  test_data_path  = data_dir + "/test.txt"

  f1=open(train_data_path,"r")
  f2=open(test_data_path,"r")

  # Read the data as a list of sentences
  train_data = [l.split() for l in f1.readlines()]
  test_data  = [l.split() for l in f2.readlines()]

  return train_data,test_data

class language_model(object):

  def __init__(self, train_data, n, laplace=1,unk_cutoff=1):
        
        self.n = n
        self.laplace = laplace
        self.unk_cutoff=unk_cutoff
        self.min_len=6
        self.max_len=20
        self.tokens = self.preprocess(train_data)
        self.vocab  = FreqDist(self.tokens)
        self.masks  = list(reversed(list(product((0,1), repeat=n))))
        self.model  = self.create_model()
        


  def preprocess(self,train_data):
    
    # Generate the tokens by adding <s> and </s>

    tokens=list(flatten(pad_both_ends(sent, n=self.n) for sent in train_data))

    # Get the vocabulary from the token

    vocab = Vocabulary(tokens, unk_cutoff=self.unk_cutoff)
    
    # Add the unknown labels to the tokens
    
    return [token if vocab[token]>self.unk_cutoff else '<UNK>' for token in tokens]


  def create_model(self):
  
    if self.n==1:
      n_tokens=len(self.tokens)
      return {(unigram,): count/n_tokens for unigram,count in self.vocab.items()}
  
  
    vocab_size=len(self.vocab)               # Get the vocab size
    n_grams=ngrams(self.tokens,self.n)       # Get the n_grams
    m_grams=ngrams(self.tokens,self.n-1)     # Get the n-1 grams

    n_vocab_dict=FreqDist(list(n_grams))
    m_vocab_dict=FreqDist(list(m_grams))     # Get the dict containing counts for the n_grams
  

    def laplace_smooth(ngram,count):
       
        mgram=ngram[:-1]
        count_mgram=m_vocab_dict[mgram]
        return (count+self.laplace)/(count_mgram +self.laplace*vocab_size)


    return {ngram: laplace_smooth(ngram,count) for ngram,count in n_vocab_dict.items()}

  
  def convert(self,ngram):
  
  #Converts the n_gram to one known by the train corpus (by trying <UNK> for each word in the ngram) 

    mask = lambda ngram, bitmask: tuple((token if flag == 1 else "<UNK>" for token,flag in zip(ngram, bitmask)))
    ngram = (ngram,) if type(ngram) is str else ngram
    for possible_known in [mask(ngram, bitmask) for bitmask in self.masks]:
        if possible_known in self.model:
            return possible_known

    return  ('FLAG')      


  def perplexity(self,test_data):

    # Calculate the perplexity function for the test data

    # Preprocess the test data
    test_tokens=self.preprocess(test_data)
    #print(test_tokens[0:20])

    # Get the n-grams of the test data
    test_ngrams=list(ngrams(test_tokens,self.n))
    #print(test_ngrams[0:20])


    # Get the vocab for the ngrams
    test_vocab = FreqDist(test_ngrams)


    # Number of tokens
    N=len(test_tokens) 
    #print("No of tokens",N)          

    # Convert to known n_grams if not found in train
    known_ngrams=[self.convert(ngram) for ngram in test_ngrams]   
    #print("No of known_ngrams",len(known_ngrams))

    known_ngrams=[i for i in known_ngrams if i is not None]

    # Get the conditional probabilites for each token
    probs =[]

    for ngram in known_ngrams:
      if ngram=='FLAG':
        probs.append(self.laplace/(self.laplace * len(self.vocab)))
      else:  
        probs.append(self.model[ngram])


    # Return the perplexity
    return math.exp((-1/(N)) * sum(map(math.log, probs)))           

  def best_candidate_word(self, prev, i, words_already_used=[]):
        
        words_already_used  = ["<UNK>"] + words_already_used
        candidates = ((ngram[-1],prob) for ngram,prob in self.model.items() if ngram[:-1]==prev)
        candidates = filter(lambda candidate: candidate[0] not in words_already_used, candidates)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return ("</s>", 1)
        else:
            return candidates[0 if prev != () and prev[-1] != "<s>" else i]
   
  def generate_sentences(self,n_sentences):
        for i in range(n_sentences):
            sentence, prob = ["<s>"] * max(1,self.n-1), 1
            while sentence[-1] != "</s>":
                
              # For n=1 we don't need prev otherwise extract prev n-1 words
                prev_n_words = () if self.n == 1 else tuple(sentence[-(self.n-1):])   
              
              # Blacklist prevents word repetition and makes the sentences have a min_len
                words_already_used = sentence + (["</s>"] if len(sentence) < self.min_len else [])

                next_word, next_prob = self.best_candidate_word(prev_n_words, i, words_already_used)
                sentence.append(next_word)
                prob *= next_prob
                
                if len(sentence) >= self.max_len:
                    sentence.append("</s>")

            yield ' '.join(sentence), -1/math.log(prob)

# The main function
if __name__ == '__main__':
  
  data_dir="/content/drive/MyDrive/lm-data"
  
  train_data,test_data=load_data(data_dir)

  # Initialize the language_model object from the train data

  lm = language_model(train_data,5)          # 4-gram language model object                    

  print("Vocabulary size: {}".format(len(lm.vocab)))

  print("Generating sentences...")
  for sentence, prob in lm.generate_sentences(10):   # Generate 10 sentences
        print("{} ({:.5f})".format(sentence, prob))
  
  perplexity=lm.perplexity(test_data)
  print("Perplexity on test data",perplexity)

