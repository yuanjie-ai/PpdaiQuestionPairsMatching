import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
tqdm.pandas('...')

import pickle


class Pickle(object):
    """
    https://blog.csdn.net/justin18chan/article/details/78516452
    json，用于字符串 和 python数据类型间进行转换:
        json只能处理简单的数据类型，比如字典，列表等，不支持复杂数据类型，如类等数据类型。
    """

    def __init__(self):
        pass

    def serialize(self, object_, file):
        """
        :param object_: python object
        :param file:
        """
        with open(file, 'wb') as f:
            pickle.dump(object_, f)

    def deserialize(self, file):
        with open(file, 'rb') as f:
            return pickle.load(f)
        
def data_reshape():
    train = pd.read_csv('./paipai/train.csv')
    test = pd.read_csv('./paipai/test.csv')
    q = pd.read_csv('./paipai/question.csv')
    func = lambda data: data.merge(q.rename(columns={'qid': 'q1'}), 'left', 'q1').merge(q.rename(columns={'qid': 'q2'}), 'left', 'q2')
    train_data = func(train)
    test_data = func(test)
    
    # 基于词向量
    col_name_wv = ['label', 'words_x', 'words_y']
    func = lambda data: pd.concat((data, data.rename(columns={'words_y': 'words_x', 'words_x': 'words_y'}))).sample(frac=1)
    train_data = func(train_data[col_name_wv])
    test_data = func(test_data[col_name_wv[1:]])
    
    # 基于字向量
    col_name_cv = ['label', 'chars_x', 'chars_y']
    func = lambda data: pd.concat((data, data.rename(columns={'chars_y': 'chars_x', 'chars_x': 'chars_y'}))).sample(frac=1)
    train_data = func(train_data[col_name_cv])
    test_data = func(test_data[col_name_cv[1:]])
    


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class KerasBow(object):
    """doc
    词袋模型：我们可以为数据集中的所有单词制作一张词表，然后将每个单词和一个唯一的索引关联。
    每个句子都是由一串数字组成，这串数字是词表中的独立单词对应的个数。
    通过列表中的索引，我们可以统计出句子中某个单词出现的次数。
    """

    def __init__(self, num_words=20000, maxlen=None):
        """
        :param maxlen: 句子序列最大长度
        :param num_words: top num_words-1(词频降序)：保留最常见的num_words-1词
        """
        self.maxlen = maxlen
        self.num_words = num_words

    def fit(self, docs):
        """
        :param corpus: ['some thing to do', 'some thing to drink']与sklearn提取文本特征一致
        """
        print('Create Bag Of Words ...')
        self.tokenizer = Tokenizer(self.num_words, lower=False) # 不改变大小写（需提前预处理）
        self.tokenizer.fit_on_texts(docs)
        print(f"Get Unique Words In Corpus: {len(self.tokenizer.word_index)}")
        # self.tokenizer.word_index
        # self.tokenizer.word_counts

    def transform(self, docs):
        print('Docs To Sequences ...')
        sequences = self.tokenizer.texts_to_sequences(docs)
        pad_docs = pad_sequences(sequences, maxlen=self.maxlen, padding='post')
        if self.maxlen is None:
            self.maxlen = pad_docs.shape[1]
        return pad_docs

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)
    
import numpy as np
from keras.layers import Embedding


class KerasEmbedding(object):
    def __init__(self, fname, maxlen, word_index):
        """
        :param fname: 词向量路径
        :param maxlen: 句序列最大长度
        :param word_index: KerasBow().tokenizer.word_index
        """
        self.fname = fname
        self.maxlen = maxlen
        self.word_index = word_index
        self.embeddings_index, self.embeddings_dim = self.gensim_load_wv()  # self.file_load_wv

    def get_keras_embedding(self, train_embeddings=False):
        print('Get Keras Embedding Layer ...')
        # prepare embedding matrix
        num_words = len(self.word_index) + 1  # 未出现的词标记0
        embedding_matrix = np.zeros((num_words, self.embeddings_dim))
        # embedding_matrix = np.random.random((num_words, EMBEDDING_DIM))  # np.random.normal(size=(num_words, EMBEDDING_DIM))
        for word, idx in self.word_index.items():
            if word in self.embeddings_index:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[idx] = self.embeddings_index[word]
        self.embedding_matrix = embedding_matrix
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(input_dim=num_words,
                                    output_dim=self.embeddings_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.maxlen,
                                    trainable=train_embeddings)
        return embedding_layer

    def gensim_load_wv(self):
        """大写会变成小写"""
        try:
            import gensim
            print('Load Word Vectors ...')
            model = gensim.models.KeyedVectors.load_word2vec_format(self.fname)
            return model, model.vector_size
        except ImportError:
            raise ImportError("Please install gensim")

    def file_load_wv(self):
        print('Load Word Vectors ...')
        embeddings_index = {}
        with open(self.fname) as f:
            for line in f:
                line = line.split()
                if len(line) > 2:
                    embeddings_index[line[0]] = np.asarray(line[1:], dtype='float32')
        return embeddings_index, len(line[1:])

kb = KerasBow(20000)
kb.fit(q.words)
bow_x = kb.transform(data_ws.words_x)
bow_y = kb.transform(data_ws.words_y)

ke = KerasEmbedding('./paipai/word_embed.txt', kb.maxlen, kb.tokenizer.word_index)
embedding_layer = ke.get_keras_embedding()




import keras
from keras.layers import Input, LSTM, GRU, Embedding, Dropout, Dense, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, Adadelta, RMSprop, Nadam

shared_lstm = LSTM(256, dropout=0.2, recurrent_dropout=0.2)
# shared_lstm = GRU(256, dropout=0.2, recurrent_dropout=0.2)

input_1 = Input(shape=(None,))
embedding_1 = embedding_layer(input_1)
encoded_1 = shared_lstm(embedding_1)

input_2 = Input(shape=(None,))
embedding_2 = embedding_layer(input_2)
encoded_2 = shared_lstm(embedding_2)


# We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([encoded_1, encoded_2])
merged_vector = Dropout(0.5)(merged_vector)
merged_vector = BatchNormalization()(merged_vector)
# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[input_1, input_2], outputs=predictions)
model.summary()

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['acc'])

model.fit([bow_x, bow_y], data_ws.label.values.reshape(-1, 1), batch_size=1280, epochs=25, validation_split=0.25)
