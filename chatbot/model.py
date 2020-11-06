import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class JustRespond:

    def __init__(self):
        self.index = 0
        self.responses = ["I don't know what to say...",
                          "Oops I have issues here, one moment",
                          "Let's start all over again",
                          "Mmmmm..."]

    def respond(self):
        ret = self.responses[self.index]
        if self.index < len(self.responses):
            self.index += 1
        else:
            self.index = 0
        return ret

class Model:
    
    MAX_LEN = 20

    def __init__(self, model_name='Seq2seqchat.h5'):
        
        self.model = tf.keras.models.load_model(model_name)
        #print("Model loaded: \n")
        self.model.summary()

        with open('tokenizer_object.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open('index2word_dict.pickle', 'rb') as handle:
            self.index2word = pickle.load(handle)

        self.encoder_sequence_1_data, self.decoder_sequence_1_data = None, None
        self.voc = ["", "bos", "eos"]
        self.voc += pickle.load(open('voc_dictionary.pickle', 'rb')) #[:10000]
        for index, item in enumerate(self.voc):
            if "bos" == item:
                print(index)
            if "eos" == item:
                print(index)

        self.jr = JustRespond()


    def enc_dec_sentence(self, sentence: str):
        encoder_sequence_1 = self.tokenizer.texts_to_sequences([sentence])
        decoder_sequence_1 = self.tokenizer.texts_to_sequences([""])

        self.encoder_sequence_1_data = pad_sequences(encoder_sequence_1, maxlen=self.MAX_LEN, dtype='int32', padding='post', truncating='post')
        #self.decoder_sequence_1_data = pad_sequences(decoder_sequence_1, maxlen=self.MAX_LEN, dtype='int32', padding='post', truncating='post')
        self.decoder_sequence_1_data = np.zeros((1, self.MAX_LEN))
        self.decoder_sequence_1_data[0, 0] = 2
        #self.decoder_sequence_1_data[0, -2] = 3
        #self.decoder_sequence_1_data[0, -1] = 1
        print("encoder_sequence_1_data: {} \n\n".format(self.encoder_sequence_1_data))
        print("decoder_sequence_1_data: {} \n\n".format(self.decoder_sequence_1_data))
        print("length check encoder[{}], decoder[{}], verdict: {}".format(len(self.encoder_sequence_1_data[0]), len(self.decoder_sequence_1_data[0]), (len(self.encoder_sequence_1_data[0]) == len(self.decoder_sequence_1_data[0]))))

    def predict(self, sentence: str):
        
        self.enc_dec_sentence(sentence)
        try:
            p = self.model.predict([self.encoder_sequence_1_data,
                                    self.decoder_sequence_1_data])
        except:
            return self.jr.respond()

        print("[predict] p={}".format(p))
        #print("[predict] argmax(p)={}".format(np.argmax(p)))
        out_index_sentence = []
        for item in p:
            for l1 in item:
                out_index_sentence.append(np.argmax(l1))
        
        print("[predict] out_index_sentence={}".format(out_index_sentence))

        out_str_sentence = ""
        for index in out_index_sentence:
            if index != 0:
                out_str_sentence += " " + self.index2word[index]

        print("[predict] out_str_sentence={}".format(out_str_sentence))

        return out_str_sentence

    def print_result(self, input):
        maxlen_input = 20
        encoder_input = self.tokenizer.texts_to_sequences([input])
        dictionary_size = len(self.voc)
        encoder_input_pad = pad_sequences(encoder_input, maxlen=self.MAX_LEN, dtype='int32', padding='post',
                                          truncating='post')
        ans_partial = np.zeros((1, maxlen_input))
        ans_partial[0, -1] = 2  # the index of the symbol BOS (begin of sentence)
        for k in range(maxlen_input - 1):
            try:
                ye = self.model.predict([encoder_input_pad, ans_partial])
                mp = np.argmax(ye)
            except:
                mp = 0
            ans_partial[0, 0:-1] = ans_partial[0, 1:]
            ans_partial[0, -1] = mp
        text = ''
        for k in ans_partial[0]:
            k = k.astype(int)
            if k < (dictionary_size - 2):
                w = self.voc[k]
                text = text + w + ' '
        return text

#if __name__ == "__main__":
    #model = Model()
    #print(model.predict("who are you"))