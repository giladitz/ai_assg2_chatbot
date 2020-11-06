import re
import codecs
import numpy as np
import pandas as pd
import string
import pickle
import operator
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model


get_ipython().run_line_magic('matplotlib', 'inline')
MAX_LEN = 20
BATCH_SIZE = 32
EPOCHS = 2
MAXIMUM_CONVERSATIONS = 304714
REDUCED_CONVERSATIONS = 30000
MAX_LEN = 12

def pp_dict(dict, items: int):
    cnt = 0
    for k, v in dict.items():
        print(k, ", ", v)
        cnt += 1
        if cnt == items:
            break

with codecs.open("movie_lines.txt", "rb", encoding="utf-8", errors="ignore") as f:
    lines = f.read().split("\n")
    conversations = []
    for line in lines:
        data = line.split(" +++$+++ ")
        conversations.append(data)

conversations = conversations[:REDUCED_CONVERSATIONS]
len(conversations)

chats = {}
for tokens in conversations:
    if len(tokens) > 4:
        idx = tokens[0][1:]
        chat = tokens[4]
        chats[int(idx)] = chat

pp_dict(chats, 10)
len(chats)

sorted_chats = sorted(chats.items(), key = lambda x: x[0])
sorted_chats[:10]

conves_dict = {}
counter = 1
conves_ids = []
for i in range(1, len(sorted_chats)+1):
    if i < len(sorted_chats):
        if (sorted_chats[i][0] - sorted_chats[i-1][0]) == 1:
            if sorted_chats[i-1][1] not in conves_ids:
                conves_ids.append(sorted_chats[i-1][1])
            conves_ids.append(sorted_chats[i][1])
        elif (sorted_chats[i][0] - sorted_chats[i-1][0]) > 1:            
            conves_dict[counter] = conves_ids
            conves_ids = []
        counter += 1
    else:
        pass

context_and_target = []
for conves in conves_dict.values():
    if len(conves) % 2 != 0:
        conves = conves[:-1]
    for i in range(0, len(conves), 2):
        context_and_target.append((conves[i], conves[i+1]))

context, target = zip(*context_and_target)

context = list(context)
target = list(target)

def clean_text(text):

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"  ", " ", text)
    text = text.strip()
    
    return text

tidy_target = []
for conve in target:
    text = clean_text(conve)
    tidy_target.append(text)

tidy_context = []
for conve in context:
    text = clean_text(conve)
    tidy_context.append(text)

bos = "<BOS> "
eos = " <EOS>"
final_target = [bos + conve + eos for conve in tidy_target] 
encoder_inputs = tidy_context
decoder_inputs = final_target
len(encoder_inputs)

"""
with open('encoder_inputs.pickle', 'wb') as handle:
    pickle.dump(encoder_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('decoder_inputs.pickle', 'wb') as handle:
    pickle.dump(decoder_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

import codecs
with codecs.open("encoder_inputs.txt", "rb", encoding="utf-8", errors="ignore") as f:
    lines = f.read().split("\n")
    encoder_text = []
    for line in lines:
        data = line.split("\n")[0]
        encoder_text.append(data)

with codecs.open("decoder_inputs.txt", "rb", encoding="utf-8", errors="ignore") as f:
    lines = f.read().split("\n")
    decoder_text = []
    for line in lines:
        data = line.split("\n")[0]
        decoder_text.append(data)

dictionary = []
for text in full_inputs:
    words = text.split()
    for i in range(0, len(words)):
        if words[i] not in dictionary:
            dictionary.append(words[i])

from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 14999
tokenizer = Tokenizer(num_words=VOCAB_SIZE)

full_text = encoder_text + decoder_text
full_inputs = encoder_inputs + decoder_inputs
okenizer.fit_on_texts(full_text)
word_index = tokenizer.word_index

tokenizer.fit_on_texts(full_inputs)
word_index = tokenizer.word_index
pp_dict(word_index, 10)
print("len word_index: ", len(word_index))

index2word = {}
for k, v in word_index.items():
    if v < 90000:
        index2word[v] = k
    if v > 90000:
        continue

pp_dict(index2word, 10)

"""
with open('index2word_dict.pickle', 'wb') as handle:
    pickle.dump(index2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

word2index = {}
for k, v in index2word.items():
    word2index[v] = k

pp_dict(word2index, 8)

"""
with open('word2index_dict.pickle', 'wb') as handle:
    pickle.dump(word2index, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

len(word2index) == len(index2word)

encoder_text = encoder_text[:10000]
decoder_text = decoder_text[:10000]

encoder_sequences_ins = tokenizer.texts_to_sequences(encoder_inputs)

decoder_sequences_ins = tokenizer.texts_to_sequences(decoder_inputs)

for seqs in encoder_sequences_ins:
    for seq in seqs:
        if seq > 14999:
            print(seq)
            break

VOCAB_SIZE = len(index2word) + 1
VOCAB_SIZE

num_samples = len(encoder_sequences_ins)
decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")

for i, seqs in enumerate(decoder_input_data):
    for j, seq in enumerate(seqs):
        if j > 0:
            decoder_output_data[i][j][seq] = 1.


encoder_input_data = pad_sequences(encoder_sequences_ins, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
decoder_input_data = pad_sequences(decoder_sequences_ins, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')

embeddings_index = {}
with open('glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

embedding_dimention = 50
def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

embedding_matrix = embedding_matrix_creater(50, word_index=word2index)


embed_layer = Embedding(input_dim=VOCAB_SIZE, output_dim=50, trainable=True,)
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix])

def seq2seq_model_builder(HIDDEN_DIM=300):
    
    encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    encoder_embedding = embed_layer(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
    decoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    decoder_embedding = embed_layer(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    
    outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    
    return model

model = seq2seq_model_builder(HIDDEN_DIM=300)
model.summary()
model.compile(optimizer='adam', loss ='categorical_crossentropy', metrics = ['accuracy'])

"""
with open('encoder_input_data.pickle', 'wb') as handle:
    pickle.dump(encoder_input_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('decoder_input_data.pickle', 'wb') as handle:
    pickle.dump(decoder_input_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('decoder_output_data.pickle', 'wb') as handle:
    pickle.dump(decoder_output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

history = model.fit([encoder_input_data, decoder_input_data], 
                     decoder_output_data, 
                     epochs=EPOCHS, 
                     batch_size=BATCH_SIZE)

model.save_weights('seq2seq_weights.h5')

test1 = tokenizer.texts_to_sequences(["hello mister gusmin"])
test1 = pad_sequences(test1, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
test2 = np.zeros((1, MAX_LEN))
for i in range(20):
    test2[0, i] = float(i)
print(test1)
print(test2)

test3 = tokenizer.texts_to_sequences(["hello to you mister gusmin?"])
test3 = pad_sequences(test3, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
test4 = np.zeros((1, MAX_LEN))
print(test3)
print(test4)

res2 = model.predict([test3, test4])
res = model.predict([test1, test2])

voc = ["", "bos", "eos"]
voc += pickle.load(open('voc_dictionary.pickle', 'rb'))
def _predict(input):
    maxlen_input = 20
    encoder_input = tokenizer.texts_to_sequences([input])
    dictionary_size = len(voc)
    encoder_input_pad = pad_sequences(encoder_input, maxlen=20, dtype='int32', padding='post',
                                      truncating='post')
    ans_partial = np.zeros((1, maxlen_input))
    for k in range(maxlen_input - 1):
        try:
            ye = model.predict([encoder_input_pad, ans_partial])
            mp = np.argmax(ye)
        except:
            mp = 0
        ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, -1] = mp
    text = ''
    for k in ans_partial[0]:
        k = k.astype(int)
        if k < (dictionary_size - 2):
            w = voc[k]
            text = text + w + ' '
    return text

ret = _predict("hello gusmin")

out_index_sentence = []
for item in res:
    for l1 in item:
        out_index_sentence.append(np.argmax(l1))

out_str_sentence = ""
for index in out_index_sentence:
    if index != 0:
        out_str_sentence += " " + index2word[index]
out_str_sentence

new_d = encoder_input_data[:20, :1]
print(new_d.shape)
print(new_d)
model.save("full_chatbot_model.h5")
