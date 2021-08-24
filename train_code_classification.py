__author__ = 'unique'

import datetime
import itertools
import os
from time import time

import keras as krs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from gensim.models import KeyedVectors
from keras import backend as K
from keras.callbacks import Callback
#from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta,Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
#from lib.data.Tree import *
from tensorflow.python.keras.layers import Lambda

from lib.transformer.transformer_classifer import TransformerBlock, TokenAndPositionEmbedding
from pycparser import c_parser
from tree import trans_to_sequences
from sklearn.metrics import classification_report,cohen_kappa_score
from keras_self_attention  import SeqSelfAttention
from transformers import BertTokenizer


TRAIN_CSV = '/codeclone_data/code_classification_data_for_Ccode.csv'

SOURCE_CODE_EMBEDDING_FILE = '/codeclone_data/train_xe_bin.abcc.code.gz'
FIG_SAVING_DIR='/codeclone_data/figures/'
source_code_embedding_file = os.path.join(os.getcwd(), *SOURCE_CODE_EMBEDDING_FILE.split('/'))

word2vec = KeyedVectors.load_word2vec_format(source_code_embedding_file,binary=True)

# Load training and test set
train_csv = os.path.join(os.getcwd(), *TRAIN_CSV.split('/'))
train_df =pd.read_csv(train_csv)


#stops = set(stopwords.words('english'))


samples_number = 30000

validation_percent = 0.2
test_percent = 0.2

df = train_df
train_df = df

parser = c_parser.CParser()
max_lenth=700
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding

code_clones_cols = ['code']
errorrows=[]
# Iterate over the questions only of both training and test datasets
for dataset in [train_df]:
    for index, row in dataset.iterrows():
        try:
            for code_clone in code_clones_cols:
                q2n = []
                ast = parser.parse(row[code_clone])
                ast = trans_to_sequences(ast)
                tokens_for_sents = ['CLS']+ast+['SEP']
                if len(tokens_for_sents) > max_lenth:
                    errorrows.append(index)
                    continue
                for word in tokens_for_sents:

                    # Check for unwanted words
                    # if word in stops and word not in word2vec.wv.vocab:
                    #     continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])
                dataset.at[index,code_clone]=q2n
        except Exception as e:
            print (e)
            errorrows.append(index)
            continue


train_df=train_df.drop(errorrows)  # remove unparsable code from training data
embedding_dim = 512
embedding_matrix = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embedding_matrix[0] = 0  # So that the padding will be ignored

max_seq_length = train_df.code.map(lambda x: len(x)).max()
print('max_seq_length: %d'% max_seq_length)
# Split to train validation
validation_size = int(len(train_df) * validation_percent)
training_size = len(train_df) - validation_size

X = train_df[code_clones_cols]
Y = train_df['label']


X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.code,
           }
X_validation = {'left': X_validation.code,

                }
X_test = {'left': X_test.code, }



# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values
Y_test = Y_test.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation,X_test], ['left']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
# assert X_train['left'].shape == X_train['right'].shape
# assert len(X_train['left']) == len(Y_train)

# Model variables
batch_size = 32
n_epoch = 40
num_heads = 8
ff_dim = 512
block_number =1


print("batch size: %d num_heads: %d ff_dim: %d block_number:%d" % (batch_size,num_heads,ff_dim,block_number))


# The visible layer
data_input = Input(shape=(max_seq_length,), dtype='int32')

vocab_size=len(embedding_matrix)
embedding_layer = TokenAndPositionEmbedding(max_seq_length, vocab_size, embedding_dim)#, embedding_matrix)
transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim)

x = embedding_layer(data_input)
for i in range(block_number):
    x = transformer_block(x)

x = Lambda(lambda x: x[:,0 ])(x)

x = layers.Dropout(0.1)(x)

encoded_x = x
classify = Dense(104,activation='softmax')(encoded_x)
# Pack it all up into a model
malstm = Model(inputs=[data_input], outputs=[classify])


optimizer=Adam(1e-5)#Adam(learning_rate=)


malstm.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Start training
training_start_time = time()

metrics = Metrics()
malstm.summary()

model = Model(inputs=malstm.input,
              outputs=[malstm.output, malstm.layers[1].output])


malstm_trained = malstm.fit(X_train['left'],  krs.utils.to_categorical(Y_train,104), batch_size=batch_size, epochs=n_epoch,
                            validation_data=(X_validation['left'],   krs.utils.to_categorical(Y_validation,104)),
                            #callbacks=[checkpointer]
                            )

outs = malstm.predict(X_train['left'][0])

print("Training time finished.\n{} epochs in {}".format(n_epoch,
                                                        datetime.timedelta(seconds=time() - training_start_time)))
scores = malstm.evaluate(X_test['left'],  krs.utils.to_categorical(Y_test,104))
print("%s: %.2f%%" % (malstm.metrics_names[1], scores[1] * 100))
for i in range(len(malstm.metrics_names)):
    print("%s: %.2f%%" % (malstm.metrics_names[i], scores[i]))


y_pred = malstm.predict(X_test['left'])
#
print(cohen_kappa_score(Y_test, np.argmax(y_pred,1)))
malstm.summary()
# Plot accuracy
plt.plot(malstm_trained.history['accuracy'])
plt.plot(malstm_trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'acc.png')


# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'loss.png')




