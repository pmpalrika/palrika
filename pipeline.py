import nltk
import pandas as pb

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler



def datapre1(data):
    ref = data.drop(
        ['id', 'created_at', 'updated_at', 'latitude', 'links', 'longitude', 'recommendation', 'user_risklevel',
         'storydate', 'validitydate', 'user_regiontype', 'user_status', 'liveevent', 'riskalert', 'background2',
         'description2', 'impact2', 'recommendation2', 'eventdate', 'impact_radius', 'importance'], axis=1)
    X = ref.drop(columns=["user_riskcategory"])
    # Y = ref["user_riskcategory"]
    x_1 = X["description"].fillna(" ")
    x_2 = X["background"].fillna(" ")
    x_3 = X["impact"].fillna(" ")
    x_4 = X["title"].fillna(" ")
    x_1 = x_1.values
    x_2 = x_2.values
    x_3 = x_3.values
    x_4 = x_4.values
    return x_1, x_2, x_3, x_4


def stemming1(x_1, x_2, x_3, x_4):
    porter = PorterStemmer()
    # lancaster=LancasterStemmer()
    for i in range(len(x_1)):
        x_1[i] = porter.stem(x_1[i])
        # print(x_1[i])
    for i in range(len(x_2)):
        x_2[i] = porter.stem(x_2[i])
        # print(p2)
    for i in range(len(x_3)):
        x_3[i] = porter.stem(x_3[i])
        # print(p3)
    for i in range(len(x_4)):
        x_4[i] = porter.stem(x_4[i])
        # print(p4)

    return x_1, x_2, x_3, x_4


def tokenization1(x_1, x_2, x_3, x_4):
    stop_words = set(stopwords.words('english'))

    for i in range(len(x_1)):
        tokens = word_tokenize(x_1[i])
        # tokens = remove_non_ascii(tokens)
        # tokens = replace
        tokens = [w for w in tokens if not w in stop_words]
        x_1[i] = ' '.join(map(str, tokens))
    #     print(x_1[i])
    for i in range(len(x_2)):
        tokens = word_tokenize(x_2[i])
        # tokens = remove_non_ascii(tokens)
        # tokens = remove_punctuation(tokens)
        tokens = [w for w in tokens if not w in stop_words]
        x_2[i] = ' '.join(map(str, tokens))
    #     print(x_2[i])

    for i in range(len(x_3)):
        tokens = word_tokenize(x_3[i])
        # tokens = remove_non_ascii(tokens)
        # tokens = remove_punctuation(tokens)
        tokens = [w for w in tokens if not w in stop_words]
        x_3[i] = ' '.join(map(str, tokens))
    #     print(x_3[i])

    for i in range(len(x_4)):
        tokens = word_tokenize(x_4[i])
        # tokens = remove_non_ascii(tokens)
        # tokens = remove_punctuation(tokens)
        tokens = [w for w in tokens if not w in stop_words]
        x_4[i] = ' '.join(map(str, tokens))
    #     print(x_4[i])
    return x_1, x_2, x_3, x_4


def vectorization1(x_1, x_2, x_3, x_4):
    vect = TfidfVectorizer()
    x1_c = vect.fit_transform(x_1)
    x2_c = vect.fit_transform(x_2)
    x3_c = vect.fit_transform(x_3)
    x4_c = vect.fit_transform(x_4)

    tf1 = TfidfTransformer(use_idf=False).fit(x1_c)
    x1_tf = tf1.transform(x1_c)

    tf2 = TfidfTransformer(use_idf=False).fit(x2_c)
    x2_tf = tf2.transform(x2_c)

    tf3 = TfidfTransformer(use_idf=False).fit(x3_c)
    x3_tf = tf3.transform(x3_c)

    tf4 = TfidfTransformer(use_idf=False).fit(x4_c)
    x4_tf = tf4.transform(x4_c)
    return x1_tf, x2_tf, x3_tf, x4_tf


def Truncate1(x1_tf, x2_tf, x3_tf, x4_tf):
    tsvd = TruncatedSVD(n_components=100)
    x1_tf_tsvd = tsvd.fit(x1_tf).transform(x1_tf)
    x2_tf_tsvd = tsvd.fit(x2_tf).transform(x2_tf)
    x3_tf_tsvd = tsvd.fit(x3_tf).transform(x3_tf)
    x4_tf_tsvd = tsvd.fit(x4_tf).transform(x4_tf)
    return x1_tf_tsvd, x2_tf_tsvd, x3_tf_tsvd, x4_tf_tsvd


def Scaler1(x1_tf_tsvd, x2_tf_tsvd, x3_tf_tsvd, x4_tf_tsvd):
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.fit(x1_tf_tsvd)
    x1_tf_tsvd = scaler.fit_transform(x1_tf_tsvd)
    # scaler.fit(x2_tf_tsvd)
    x2_tf_tsvd = scaler.fit_transform(x2_tf_tsvd)
    # scaler.fit(x3_tf_tsvd)
    x3_tf_tsvd = scaler.fit_transform(x3_tf_tsvd)
    # scaler.fit(x4_tf_tsvd)
    x4_tf_tsvd = scaler.fit_transform(x4_tf_tsvd)

    return x1_tf_tsvd, x2_tf_tsvd, x3_tf_tsvd, x4_tf_tsvd
    # X.shape
    # return X,Y


def fin1(x1_tf_tsvd, x2_tf_tsvd, x3_tf_tsvd, x4_tf_tsvd):
    x1_df = pb.DataFrame(x1_tf_tsvd)
    x2_df = pb.DataFrame(x2_tf_tsvd)
    x3_df = pb.DataFrame(x3_tf_tsvd)
    x4_df = pb.DataFrame(x4_tf_tsvd)

    X = pb.DataFrame()
    X = pb.concat([x1_df, x2_df, x3_df, x4_df], axis=1, sort=False)
    return X


def process_data(data):

    data = pb.DataFrame(data, index=[0])

    print(data)


    x_1, x_2, x_3, x_4 = datapre1(data)
    x_1, x_2, x_3, x_4 = stemming1(x_1, x_2, x_3, x_4)
    x_1, x_2, x_3, x_4 = tokenization1(x_1, x_2, x_3, x_4)
    x1_tf, x2_tf, x3_tf, x4_tf = vectorization1(x_1, x_2, x_3, x_4)

    # x1_tf_tsvd, x2_tf_tsvd, x3_tf_tsvd, x4_tf_tsvd = Truncate1(x1_tf, x2_tf, x3_tf, x4_tf)
    # x1_tf_tsvd, x2_tf_tsvd, x3_tf_tsvd, x4_tf_tsvd = Scaler1(x1_tf_tsvd, x2_tf_tsvd, x3_tf_tsvd, x4_tf_tsvd)

    # X = fin1(x1_tf_tsvd, x2_tf_tsvd, x3_tf_tsvd, x4_tf_tsvd)
    #
    # return X

    X = fin1(x1_tf, x2_tf, x3_tf, x4_tf)
    return X

    return "asd"
