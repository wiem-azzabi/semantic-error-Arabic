# =========================
# ==== Helper Methods =====
import qalsadi.lemmatizer
from nltk.stem.isri import ISRIStemmer
import re
import numpy as np
import nltk
from nltk import ngrams
from nltk.corpus import stopwords


def clean_str(text):
    search = ["اً", "أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ",
              '"', "ـ", "'", "ى", "\\", '\n', '\t', '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ا", "ة", " ", " ", "", "", "", " و",
               " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ', ' ! ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    text = text.replace('فف', 'ف')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim
    text = text.strip()

    return text


def stringify(list):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in list:
        str1 += " "
        str1 += ele
    # return string
    return str1


def clean_stopwords(text):
    stop_words = set(stopwords.words('arabic'))
    words = nltk.word_tokenize(text)
    tokens = [token for token in words if token not in stop_words]
    text = stringify(tokens)
    return(text)


def very_clean(text):
    s = clean_str(clean_stopwords(text))
    return s


# -- Preprocessing

# lemmatization
#! pip install qalsadi


def lemmatize(text):
    lemmer = qalsadi.lemmatizer.Lemmatizer()
    t = lemmer.lemmatize_text(text)
    s = stringify(t)
    return s


def stemmatize(text):
    stemmer = ISRIStemmer()
    tokens = text.split()
    stem_tokens = [stemmer.suf32(token) for token in tokens]
    s = stringify(stem_tokens)
    return(s)

# -- error verif functions


def OneSubstitution(word, word2):  # return true if exists one and only onesubstituion error
    if len(word) == len(word2):
        l = []
        for i in range(len(word)):
            if word[i] == word2[i]:
                l.append(0)
            else:
                l.append(1)
        return (sum(l) == 1)
    else:
        return(False)


def OneInversion(word, word2):
    if len(word) == len(word2):  # returns true if exists one and only inversion error
        l1 = []
        l2 = []
        for i in range(len(word)):
            if word[i] == word2[i]:
                l1.append(0)
            else:
                l1.append(1)
                l2.append(word[i])
                l2.append(word2[i])
        if sum(l1) == 2:
            return(l2[0] == l2[3] and l2[1] == l2[2])
        else:
            return(False)
    else:
        return(False)


def AddSubErr(word1, word2):  # ajout/suppression d'un seul caracter
    w1 = set(word1)
    w2 = set(word2)
    maxi = max(len(w1-w2), len(w2-w1))
    if maxi != 1:
        return(False)
    else:
        if len(w1)-len(w2) == 1:
            lw1 = list(word1)
            lw1.remove(''.join(w1-w2))
            s = ''.join(str(x) for x in lw1)
            return(s == word2)
        elif len(w1)-len(w2) == -1:
            lw2 = list(word2)
            lw2.remove(''.join(w2-w1))
            s = ''.join(str(x) for x in lw2)
            return(s == word1)
        else:
            return(False)


def lookslike(words, corpustoken):
    for word in words:
        for c in corpustoken:
            if AddSubErr(word, c):
                return(word)
            elif OneInversion(word, c) or OneSubstitution(word, c):  # inversion / substitution
                return(word)
    return('')


def CheckNGramInCorpus(texttoken, corpustoken):
    ngram = [g for g in texttoken if g not in corpustoken]
    return (ngram)


# --embedding :
def get_vec(n_model, dim, token):
    vec = np.zeros(dim)
    is_vec = False
    if token not in n_model.wv:
        _count = 0
        is_vec = True
        for w in token.split("_"):
            if w in n_model.wv:
                _count += 1
                vec += n_model.wv[w]
        if _count > 0:
            vec = vec / _count
    else:
        vec = n_model.wv[token]
    return vec


def calc_vec(pos_tokens, neg_tokens, n_model, dim):
    vec = np.zeros(dim)
    for p in pos_tokens:
        vec += get_vec(n_model, dim, p)
    for n in neg_tokens:
        vec -= get_vec(n_model, dim, n)

    return vec

# -- Retrieve all ngrams for a text in between a specific range


def get_all_ngrams(text, nrange=3):
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\!\?\؟\^]', ' ', text)
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    ngs = []
    for n in range(2, nrange+1):
        ngs += [ng for ng in ngrams(tokens, n)]
    return ["_".join(ng) for ng in ngs if len(ng) > 0]

# -- Retrieve all ngrams for a text in a specific n


def get_ngrams(text, n=2):
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\!\?\؟\^]', ' ', text)
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    ngs = [ng for ng in ngrams(tokens, n)]
    return ["_".join(ng) for ng in ngs if len(ng) > 0]

# -- filter the existed tokens in a specific model


def get_existed_tokens(tokens, n_model):
    return [tok for tok in tokens if tok in n_model.wv]

# -- model evaluation


def precision(y, y_pred):
    c = 0  # nbre de mots correctement étiquetés
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            c += 1

    p = c / len(y)
    return(p)


def precision2(y, y_pred):  # si le modele retourne plusieurs resultats
    c = 0  # nbre de mots correctement étiquetés
    for i in range(len(y)):
        if y[i] in y_pred[i]:
            c += 1
    precision = c / len(y)
    return(precision)
