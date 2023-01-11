import numpy as np       # linear algebra
import pandas as pd      # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords     #stopwords
import nltk
nltk.download('stopwords')
from collections import Counter
import string
import re                             #regular expression library
import seaborn as sns                 #seaborn library
from tqdm import tqdm
import matplotlib.pyplot as plt      
from torch.utils.data import TensorDataset, DataLoader           #dataloader
from sklearn.model_selection import train_test_split

"""
CHECKING IF GPU IS AVAILABLE"""

is_cuda=torch.cuda.is_available()

if is_cuda:
  device=torch.device("cuda")
  print("GPU IS AVAILABLE")
else:
  device=torch.device("cpu")
  print("GPU NOT AVAILABLE,CPU USED")

"""LOADING THE IMDB DATASET"""

import csv
df=pd.read_csv("IMDB_Dataset.csv",quoting=csv.QUOTE_NONE)
df.head()

"""SPLITTING OF TRAINING AND TEST SET"""

X,y = df['review'].values,df['sentiment'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y)
print("The shape of train data is {}".format(X_train.shape))
print("The shape of test data is {}".format(X_test.shape))

type(y_train)

"""ANALYZING THE SENTIMENT  IN TRAINSET"""

v=pd.Series(y_train).value_counts()
print(v)
dd=v.values
plt.figure(figsize=(5,5))
sns.barplot(x=['Negative','Positive'],y=dd)
plt.title("No of Positive and Negative reviews in Train set")
plt.show()

"""PREPROCESSING A WORD TO REMOVE SPECIAL CHARACTERS AND DIGITS"""

def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)                       # Remove everything except letters,digits and spaces by no space 
    s = re.sub(r"\s+", '', s)                           # Replace all runs of whitespaces with no space
    s = re.sub(r"\d", '', s)                            # replace digits with no space
    return s

"""TOCKENIZATION OF EACH WORD IN A REVIEW 
AND REPLACE WORDS WITH ONE-HOT ENCODED VALUES IN TRAINING AND TEST SET
"""

def tockenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(stopwords.words('english')) 
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)                     
            if word not in stop_words and word != '':                                                       #REMOVING THE STOPWORDS
                word_list.append(word)
  
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
    final_corpus=[]
    # tockenize
    final_list_train,final_list_test = [],[]                                                                #ONE-HOT ENCODING THE TRAIN SET
    for sent in x_train:                                                                                    
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() if preprocess_string(word) in onehot_dict.keys()]) 
    
    for sent in x_train:
            final_corpus.append([preprocess_string(word) for word in sent.lower().split()                   #MAKING OF THE CORPUS
                                     if preprocess_string(word) in onehot_dict.keys()])                             
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()   #ONE-HOT ENCODING THE TEST SET
                                    if preprocess_string(word) in onehot_dict.keys()])
            
    encoded_train = [1 if label =='positive' else 0 for label in y_train]  
    encoded_test = [1 if label =='positive' else 0 for label in y_val] 
    return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test), np.array(encoded_test),onehot_dict,final_corpus

X_train,y_train,X_test,y_test,vocab,corpus = tockenize(X_train,y_train,X_test,y_test)

print(corpus[1])                 #corpus is the corpus of the training set

print(X_train[0])                #one hot encoded training

"""LENGTH OF VOCABULARY"""

print(f'Length of vocabulary is {len(vocab)}')

"""ANALYZING THE LENGTH OF REVIEWS"""

review_length = [len(i) for i in X_train]
length_data=pd.Series(review_length)
plt.figure(figsize=(8,6))
sns.distplot(length_data,color='red',kde_kws={"shade":True},kde=True)
plt.title("Analysis of Review Lengths in the Training Set")
plt.xlabel("Length of Review")
plt.ylabel("Proportion of Reviews")
plt.show()
length_data.describe()

"""PRINTING THE FIRST REVIEW IN THE TRAINING SET AFTER TOCKENIZATION AND ENCODING"""

print(X_train[0:3])

print(y_train)

type(X_train)

print(X_test)

print(y_test)

"""PADDING"""

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

#we have very less number of reviews with length > 500.
#So we will consideronly those below it.                             #PADDING TO LENGTH 500
x_train_pad = padding_(X_train,500)
x_test_pad = padding_(X_test,500)

"""BATCHING AND LOADING AS TENSOR"""

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

# dataloaders
batch_size = 50                 #Batchsize=50

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

"""LOOKING AT ONE SAMPLE BATCH OF TRAINING DATA"""

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Input size of a random batch of size 50 and length 500:', sample_x.size())         # batch_size, seq_length
print('Sample input: \n', sample_x)
print('Sample output: \n', sample_y)

"""TRAINING OUR CORPUS USING WORD2VEC"""

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
model1 = Word2Vec(sentences=corpus, window=5, min_count=1, workers=4)    #Embedding dimensions 100
model1.save("word2vec.model")

"""LOOKING AT A WORD EMBEDDING OF A WORD IN WORD2VEC"""

vector = model1.wv['br']
print(vector)

"""VISUALIZATION OF THE WORD2VEC MODEL

1. LOOKING AT WORDS MOST SIMILAR TO THE WORD "GOOD"
"""

model1.most_similar('good')[:5]                     #finding 5 most similar words

"""VISUALIZING THE WORD2VEC EMBEDDINGS"""

from sklearn.manifold import TSNE
v=list(model1.wv.vocab)
V=model1[v]
tsne = TSNE(n_components=2)
V_tsne = tsne.fit_transform(V[:200,:])

"""DIMENSIONALITY REDUCTION TO REPRESENT THE WORDS IN A 2D PLANE"""

vo_df = pd.DataFrame(V_tsne, index=v[:200], columns=['x', 'y'])
fig = plt.figure(figsize=(11,11))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(vo_df['x'], vo_df['y'])
for word, pos in vo_df.iterrows():                                             # plotting of the words in Word2vec embeddings 
    ax.annotate(word, pos)
ax.grid()

"""

*   As we can see in the representation above similar words have very close embeddings. For example,the words 'shows' and 'television' have very similar embeddings.Also, words like 'cast' and 'directed' have similar embeddings and so on.
*   Hence, the Word2Vec model can represent meanings of the words in our corpus.

"""

print(vocab)                    #Printing the vocabulary of our training set

type(vocab)

embedding_matrix=np.zeros((len(vocab)+1,100))
embedding_matrix[0]=0                                       #EMBEDDING MATRIX having Word2VEC embeddings trained on our corpus
for word,i in vocab.items():
  embedding_matrix[i]=model.wv[word]                        # Making sure the embeddings have correct indices as our one-hot encoding indices

print(embedding_matrix[6])                                 #Printing Word2Vec embeddings for the word 6

embe=torch.FloatTensor(embedding_matrix)                    #Creating a tensor of Word2Vec embedding matrix

print(embe)                                                 #Printing the embedding matrix tensor

"""DEFINING THE MODEL"""

class SentimentRNN(nn.Module):
    def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
        super(SentimentRNN,self).__init__()
 
        self.output_dim = output_dim                                         #OUTPUT-DIMENSIONS =1 (Probablity that it is a positive review)
        self.hidden_dim = hidden_dim                                         # hidden dim(hyperparameter no of features in h(activations))
 
        self.no_layers = no_layers                                           # NO OF LAYERS OF LSTM LAYER(STACKING) 
        self.vocab_size = vocab_size
    
        # embedding and LSTM layers
        self.embedding = nn.Embedding.from_pretrained(embe,freeze=True)      #USING THE PRE-TRAINED WORD2VEC EMBEDDING MATRIX
                                                                             #FREEZE=TRUE MEANS IT IS NOT PART OF LEARNING
        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
        
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)                                       #USING DROPOUT BETWEEN THE LSTM LAYERS TO HELP REGULARIZATION
                                                                           
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)                     #A FULLY CONNECTED NN ON TOP OF LSTM LAYER
        self.sig = nn.Sigmoid()
         
    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  
      
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
        
        
        
    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden

no_layers = 2
vocab_size = len(vocab) + 1 #extra 1 for padding
embedding_dim=100
output_dim = 1
hidden_dim = 256


model = SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)

#moving to gpu
model.to(device)

print(model)

# loss and optimization functions
lr=0.001                                                      # learning rate used

criterion = nn.BCELoss()                                      # Binary Cross Entropy loss is used

optimizer = torch.optim.Adam(model.parameters(), lr=lr)       # Using Adam optimzer for backpropagation 

# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

"""TRAINING OF THE MODEL """

clip = 5                                                       # Clipping is used to tackle problem of exploding gradients 
epochs = 7                                                     # Number of epochs
valid_loss_min = np.Inf
# train for some number of epochs
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state 
    h = model.init_hidden(batch_size)
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)   
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        
        model.zero_grad()
        output,h = model(inputs,h)
        
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        train_losses.append(loss.item())
        accuracy = acc(output,labels)
        train_acc += accuracy
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
 
    
        
    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in valid_loader:
            val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())
            
            accuracy = acc(output,labels)
            val_acc += accuracy
            
    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc/len(train_loader.dataset)
    epoch_val_acc = val_acc/len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
    if epoch_val_loss <= valid_loss_min:
        valid_loss_min = epoch_val_loss
    print(25*'==')

"""*   Above, we can see the training,test accuracy and loss function values over the epochs.

*  As we can see the final training accuracy is 90.5% and test accuracy is 86.36  which is quite good.

*  It was also observed that there was a significant improvement in the accuracy on using Word2Vec word embeddings.

VISUALIZATION OF THE TRAINING AND VALIDATION ACCURACY OVER EPOCHS
"""

fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 2, 1)
plt.plot(epoch_tr_acc, label='Train Acc')
plt.plot(epoch_vl_acc, label='Validation Acc')
plt.title("Accuracy")
plt.legend()
plt.grid()
    
plt.subplot(1, 2, 2)
plt.plot(epoch_tr_loss, label='Train loss')
plt.plot(epoch_vl_loss, label='Validation loss')
plt.title("Loss")
plt.legend()
plt.grid()

plt.show()

"""

*   We can see that the training and the validation accuracies increase over the number of epochs.
"""

def predict_text(text):
        word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
                         if preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(padding_(word_seq,500))
        inputs = pad.to(device)                                                   #FUNCTION FOR PREDICTING SENTIMENT OF A RANDOM TEXT
        batch_size = 1
        h = model.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
        return(output.item())

"""PREDICTION FOR  RANDOM TEXTS ENTERED BY ME

*   A POSTIVE REVIEW
"""

print('='*70)
text="this movie was one of the world's best ones.it had a good ending and is awesome for fun time with family and friends. The climax was entralling with us gripping onto our seats"
print(text)
print(f'Actual sentiment is  Positive')
print('='*70)
pro = predict_text(text)
status = "positive" if pro > 0.5 else "negative"
pro = (1 - pro) if status == "negative" else pro
print(f'predicted sentiment is {status} with a probability of {pro}')

"""

*   A NEGATIVE REVIEW

"""

print('='*70)
text2="What a horrible movie. It had the worst ending and the graphics was terrible. I would never watch it again"
print(text2)
print(f'Actual sentiment is  negative')
print('='*70)
pro2 = predict_text(text2)
status2 = "positive" if pro2 > 0.5 else "negative"
pro2 = (1 - pro2) if status2 == "negative" else pro2
print(f'predicted sentiment is {status2} with a probability of {pro2}')

""" A REAL REVIEW TESTED ON THE MODEL"""

print('='*70)
text3="Dutt is, in fact, both the strongest and weakest link in the film. In the climax scene, which has several actors coming together, he scores. Ray, not a lover of whodunits, thought denouements were long and boring, but here, Dutt keeps the audience hooked. Again, the biggest misfit in this complicated Sharadindu novel on misfits is Dutt himself, who could have done better by explaining why he was there at all and had Byomkesh chasing him at the end — is it an allegory for real life where Byomkesh, as a subject, keeps haunting him? "
print(text3)
print('='*70)
pro3 = predict_text(text3)
status3 = "positive" if pro3 > 0.5 else "negative"
pro3 = (1 - pro3) if status3 == "negative" else pro3
print(f'predicted sentiment is {status3} with a probability of {pro3}')

"""THE END"""

