import re
import urllib.parse
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import requests
import sys
from keras.callbacks.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from nltk.corpus import stopwords
from scipy import sparse as sp_sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

class SpiderRumor(object):
    def __init__(self):
        self.url_rolldown = "https://vp.fact.qq.com/loadmore?artnum=0&page=%s"  # url for directly rolling down the website
        self.url_search = "https://vp.fact.qq.com/searchresult?title=%s&num=%d"  # url for searching keyword
        self.header = {
            "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50"
        }

    def spider_run(self):
        df_all = list()
        df_tag = list()
        # get data by rolling down
        for url in [self.url_rolldown % i for i in range(50)]:
            try:
                data_list = requests.get(url, headers=self.header).json()["content"]
                temp_data = [[df["title"], df["date"], df["author"], df["result"], df["id"], df['tag']]
                             for df in data_list]
                df_all.extend(temp_data)
            except:
                continue
        # detach the tag information because the searching way won't give tag information
        for i in range(len(df_all)):
            df_tag.append([df_all[i][-1]])
            df_all[i] = df_all[i][0:5]

        # get data by searching the key word
        search_keyword = ['病毒', '肺炎', '口罩']
        code_keyword = []
        for word in search_keyword:
            code_keyword.append(urllib.parse.quote(word))
        for keyword in code_keyword:
            temp = requests.get(self.url_search % (keyword, 0), headers=self.header)
            total = temp.json()["total"]
            for url in [self.url_search % (keyword, i * 20) for i in range(total // 20)]:
                try:
                    data_list = requests.get(url, headers=self.header).json()["content"]
                    temp_data = [[df["_source"]["title"], df["_source"]["date"], df["_source"]["author"],
                                  df["_source"]["result"], df["_source"]["id"]]
                                 for df in data_list]
                    df_all.extend(temp_data)
                except:
                    continue
        # add the front part to id to make it an accessible url
        arturl = 'https://vp.fact.qq.com/article?id='
        for i in range(len(df_all)):
            df_all[i][-1] = arturl + df_all[i][-1]
            df_all[i][-2] = df_all[i][-2][0]
        # generate csv
        df = pd.DataFrame(df_all, columns=["title", "date", "author", "result", "id", ])
        df_t = pd.DataFrame(df_tag, columns=["tag"])
        df.drop_duplicates(subset='id', inplace=True)
        df.to_csv("冠状病毒谣言数据.csv", encoding="utf_8_sig")
        df_t.to_csv("冠状病毒谣言关键词.csv", encoding="utf_8_sig")
        return df, df_t


def SpiderTrans(data, column):  # translate the chinese into English
    for i in column:
        for n in range(data.shape[0]):
            data.iat[n, i] = trans_helper(data.iat[n, i])
            print('%d \r' % (int(i) * data.shape[0] + n))
    return data


def trans_helper(data):  # translate a single phrase using youdao
    url_trans = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    Data_list = {
        'type': "AUTO",
        'i': data,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    response = requests.post(url_trans, data=Data_list, headers={
        "User-Agent": 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50'})
    return response.json()['translateResult'][0][0]['tgt']


def classify():
    df = pd.DataFrame(pd.read_csv('rumor3.12.csv'))
    df.sort_values(axis=0, ascending=True, by='result', inplace=True)
    flag = int(0)
    f = pd.DataFrame
    t = pd.DataFrame
    s = pd.DataFrame
    for i in range(df.shape[0]):
        if (df.iat[i, -2] == 'TRUE') and flag == int(0):
            f = df.iloc[flag:i]
            flag = i
        if (df.iat[i, -2] == 'suspect') and flag != int(0):
            t = df.iloc[flag:i]
            s = df.iloc[i:df.shape[0]]
            break
    f.to_csv("false.csv", encoding="utf_8_sig")
    t.to_csv("true.csv", encoding="utf_8_sig")
    s.to_csv("suspect.csv", encoding="utf_8_sig")



def text_prepare(text):
    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])  # delete stopwords from text

    return text


def most_words(text):
    text=[text_prepare(x) for x in text]
    words_counts = {}
    list_word = []
    tokenize=nltk.tokenize.WhitespaceTokenizer()
    for sentence in text:
        list_word.extend(tokenize.tokenize(sentence))
    for word in set(list_word):
        words_counts[word] = list_word.count(word)
    words_most=sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
    return words_most, words_counts


def read_data(filename):  # read data from a file
    df = pd.DataFrame(pd.read_csv(filename))
    label = df['result'].values
    input_text = df['title'].values
    return label, input_text


def three_case():#seperate three cases
    _, t=read_data('true.csv')
    _, f = read_data('false.csv')
    _, s = read_data('suspect.csv')
    t,_=most_words(t)
    f,_=most_words(f)
    s,_=most_words(s)
    return t,f,s


def bag_of_word(string, WORDS_TO_INDEX, dict_size=5000):
    result_vector = np.zeros(dict_size)
    for word in string.split():
        if word in WORDS_TO_INDEX:
            result_vector[WORDS_TO_INDEX[word]] = 1
    return result_vector


def train_classifier(X_train, Y_train, C=1.0, penalty='l2'):
    lr=LogisticRegression(C=C,penalty=penalty)
    model=OneVsRestClassifier(lr)
    model.fit(X_train,Y_train)
    return model


def one_hot_class(penalty='l2',C=1.0,dict_size=5000,):# muticlass classifier
    Y,X=read_data('rumor3.12.csv')
    X = [text_prepare(x) for x in X]
    Y = [text_prepare(x) for x in Y]

    X_test=X[int(0.8*(len(X))):len(X)]
    X_train=X[0:int(0.8*(len(X)))]
    Y_test = np.array(Y[int(0.8 * (len(Y))):len(Y)]).reshape(-1, 1)
    Y_train = np.array(Y[0:int(0.8 * (len(Y)))]).reshape(-1, 1)

    X_count,_=most_words(X)
    _,Y_count=most_words(Y)
    WORDS_TO_INDEX = {p[0]: i for i, p in enumerate(X[:dict_size])}
    INDEX_TO_WORDS = {WORDS_TO_INDEX[k]: k for k in WORDS_TO_INDEX}
    ALL_WORDS = WORDS_TO_INDEX.keys()

    X_train_bag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(bag_of_word(text, WORDS_TO_INDEX, dict_size)) for text in X_train])
    X_test_bag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(bag_of_word(text, WORDS_TO_INDEX, dict_size)) for text in X_test])
    mlb = OneHotEncoder()
    Y_train=mlb.fit_transform(Y_train)
    Y_test = mlb.fit_transform(Y_test)


    classifier_bag=train_classifier(X_train_bag,Y_train,C=C,penalty=penalty)
    Y_test_pre_label=classifier_bag.predict(X_test_bag)
    Y_test_pre_score = classifier_bag.decision_function(X_test_bag)


    print('accuracy: %f'%accuracy_score(Y_test,Y_test_pre_label))
    print('f1 score: %f'%f1_score(Y_test,Y_test_pre_label,average='weighted'))


def lstm_model(max_features, embed_size,X):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=X.shape[1]))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2,kernel_regularizer=l2(0.03), recurrent_regularizer=l2(0.03), bias_regularizer=l2(0.03)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_fit(model, x, y):
    return model.fit(x, y, batch_size=100, epochs=20, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


def model_predict(model, x):
    return model.predict_classes(x)


def lstm():#lstm neural network
    max_features=5000
    embed_size=128
    maxlen=500
    y,x=read_data('rumor3.12.csv')
    x = [text_prepare(x) for x in x]
    y = [text_prepare(x) for x in y]
    x_test = x[int(0.8 * (len(x))):len(x)]
    x_train = x[0:int(0.8 * (len(x)))]
    y_test = y[int(0.8 * (len(y))):len(y)]
    y_train = y[0:int(0.8 * (len(y)))]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    y_train = pd.get_dummies(y_train).values
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    y_test = pd.get_dummies(y_test).values

    model = lstm_model(max_features, embed_size,x_train)
    history = model_fit(model, x_train, y_train)
    accr = model.evaluate(x_test, y_test)

    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #spider = SpiderRumor() #crawl the data
    #(df,dt_tag)=spider.spider_run()
    #classify()#seperate three kinds of news
    #one_hot_class(C=100)

    lstm()
'''#find top ten hot words
    t,f,s=three_case()
    print('top ten words in TRUE')
    print(t[:10])
    print('top ten words in FALSE' )
    print(f[:10])
    print('top ten words in suspect')
    print(s[:10])
'''


'''#translation
    csv_file='冠状病毒谣言数据.csv'#chanslate into english
    df=pd.DataFrame(pd.read_csv(csv_file))
    df_english=SpiderTrans(df,[1,3,4])
    df_english.to_csv("coronavirus_rumor.csv", encoding="utf_8_sig")
'''

'''#visualization
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #piechart
    data = pd.read_csv("冠状病毒谣言数据.csv")
    for i in range(len(data)):
        data.iat[i,4]=data.iat[i,4][0]
    labels = data["result"].value_counts().index.tolist()
    sizes = data["result"].value_counts().values.tolist()
    colors = ['lightgreen', 'gold', 'lightskyblue', 'lightcoral']
    plt.figure(figsize=(20, 8))
    plt.pie(sizes, labels=labels,
            colors=colors, autopct='%1.1f%%', shadow=True, startangle=50)
    plt.axis('equal')
    plt.show()

    #histogram
    data = pd.read_csv("冠状病毒谣言关键词.csv")
    df = pd.Series([j for i in [eval(i) for i in data["tag"].tolist()] for j in i]).value_counts()[:20]
    X = df.index.tolist()
    y = df.values.tolist()
    plt.figure(figsize=(15, 8))
    plt.bar(X, y, color="orange")
    plt.tight_layout()
    plt.grid(axis="y")
    plt.grid(ls='-.')
    plt.show()
'''
