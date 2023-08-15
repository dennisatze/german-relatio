# -*- coding: utf-8 -*-
"""
@author: dennisa
"""

import time
from collections import Counter
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import spacy
from tqdm import tqdm
import warnings

import json
import re
import string
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer


import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import gensim.downloader as api
import numpy as np
from gensim.models import Word2Vec
from numpy.linalg import norm
from sklearn.cluster import KMeans
from tqdm import tqdm
import spacy
from spacy.language import Language

nlp = spacy.load("de_core_news_lg")

# custom sentencizer

"""
def Custom_Sentencizer(doc):
    ''' Look for sentence start tokens by scanning for periods only. '''
    for i, token in enumerate(doc[:-2]):  # The last token cannot start a sentence
        if token.text == ".":
            doc[i+1].is_sent_start = True
        else:
            doc[i+1].is_sent_start = False  # Tell the default sentencizer to ignore this token
    return doc



@Language.factory('custom_sentencizer')
def custom_sentencizer(nlp, name):
    return Custom_Sentencizer()

"""


def split_into_sentences(
    dataframe: pd.DataFrame,
    output_path: Optional[str] = None,
    progress_bar: bool = False,
    text_col = "text",
    id_col = "id",
    method: Optional[str] = "nltk"
) -> Tuple[List[str], List[str]]:

    """
    A function that splits a list of documents into sentences (using the SpaCy sentence splitter).
    Args:
        dataframe: a pandas dataframe with one column "id" and one column "doc"
        output_path: path to save the output
        progress_bar: print a progress bar (default is False)
    Returns:
        Tuple with the list of document indices and list of sentences
    """

    docs = dataframe.to_dict(orient="records")

    sentences: List[str] = []
    doc_indices: List[str] = []

    if method == "nltk":
        if progress_bar:
            print("Splitting into sentences...")
            docs = tqdm(docs)

        for doc in docs:
            for sent in sent_tokenize(doc[text_col],language='german'):
                sentences.append(str(sent))
                doc_indices = doc_indices + [doc[id_col]]

        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump((doc_indices, sentences), f)



    if method == "spacy":
        if progress_bar:
            print("Splitting into sentences...")
            docs = tqdm(docs)

        for doc in docs:
            for sent in nlp(doc["doc"], disable=["tagger", "ner"]).sents:
                sentences.append(str(sent))
                doc_indices = doc_indices + [doc[id_col]]

        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump((doc_indices, sentences), f)

    return (doc_indices, sentences)




def mine_entities(
    sentences: List[str],
    ent_labels: Optional[List[str]] = ["PERSON", "NORP", "ORG", "GPE", "EVENT"],
    remove_punctuation: bool = True,
    remove_digits: bool = True,
    remove_chars: str = "",
    stop_words: Optional[List[str]] = None,
    lowercase: bool = True,
    strip: bool = True,
    remove_whitespaces: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    tags_to_keep: Optional[List[str]] = None,
    remove_n_letter_words: Optional[int] = None,
    progress_bar: bool = False,
) -> Counter:

    """
    A function that goes through sentences and counts named entities found in the corpus.
    Args:
        sentences: list of sentences
        ent_labels: list of entity labels to be considered (see SpaCy documentation)
        progress_bar: print a progress bar (default is False)
        For other arguments see utils.clean_text.
    Returns:
        Counter with the named entity and its associated frequency on the corpus
    """

    entities_all = []

    if progress_bar:
        print("Mining named entities...")
        time.sleep(1)
        sentences = tqdm(sentences)

    for sentence in sentences:
        sentence = nlp(sentence)
        for ent in sentence.ents:
            if ent.label_ in ent_labels:
                entities_all.append(ent.text)
    """
    entities_all = clean_text(
        entities_all,
        remove_punctuation,
        remove_digits,
        remove_chars,
        stop_words,
        lowercase,
        strip,
        remove_whitespaces,
        lemmatize,
        stem,
        tags_to_keep,
        remove_n_letter_words,
    )
    """
    # forgetting to remove those will break the pipeline
    entities_all = [entity for entity in entities_all if entity != ""]

    entity_counts = Counter(entities_all)

    return(entity_counts)


def compute_sif_weights(words_counter, alpha = 0.001):

    """
    A function that computes smooth inverse frequency (SIF) weights based on word frequencies.
    (See "Arora, S., Liang, Y., & Ma, T. (2016). A simple but tough-to-beat baseline for sentence embeddings.")
    Args:
        words_counter: a dictionary {"word": frequency}
        alpha: regularization parameter
    Returns:
        A dictionary {"word": SIF weight}
    """

    sif_dict = {}

    for word, count in words_counter.items():
        sif_dict[word] = alpha / (alpha + count)

    return(sif_dict)



def obtain_ner(doc):

# takes as input a doc that has run through spacy's nlp() function with
# active ner
# output: a list with tuples of (token, ner_tag)

    t = [(tok, tok.label_) for tok in doc.ents]
    return(t)


def replace_sentences(
    sentences: List[str],
    max_sentence_length: Optional[int] = None,
    min_sentence_length: Optional[int] = None,
    max_number_words: Optional[int] = None,
) -> List[str]:

    """
    Replace long sentences in list of sentences by empty strings.
    Args:
        sentences: list of sentences
        max_sentence_length: Keep only sentences with a a number of character lower or equal to max_sentence_length. For max_number_words = max_sentence_length = -1 all sentences are kept.
        max_number_words: Keep only sentences with a a number of words lower or equal to max_number_words. For max_number_words = max_sentence_length = -1 all sentences are kept.
    Returns:
        Replaced list of sentences.
    Examples:
        >>> replace_sentences(['This is a house'])
        ['This is a house']
        >>> replace_sentences(['This is a house'], max_sentence_length=15)
        ['This is a house']
        >>> replace_sentences(['This is a house'], max_sentence_length=14)
        ['']
        >>> replace_sentences(['This is a house'], max_number_words=4)
        ['This is a house']
        >>> replace_sentences(['This is a house'], max_number_words=3)
        ['']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=18)
        ['This is a house', 'It is a nice house']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=4, max_sentence_length=18)
        ['This is a house', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=17)
        ['This is a house', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=0, max_sentence_length=18)
        ['', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=0)
        ['', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'])
        ['This is a house', 'It is a nice house']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=4)
        ['This is a house', '']
    """

    if max_sentence_length is not None:
        sentences = [
            "" if (len(sent) > max_sentence_length) else sent for sent in sentences
        ]

    if min_sentence_length is not None:
            sentences = [
                "" if (len(sent) < min_sentence_length) else sent for sent in sentences
            ]

    if max_number_words is not None:
        sentences = [
            "" if (len(sent.split()) > max_number_words) else sent for sent in sentences
        ]

    return (sentences)




def count_values(
    dicts: List[Dict], keys: Optional[list] = None, progress_bar: bool = False
) -> Counter:

    """
    Get a counter with the values of a list of dictionaries, with the conssidered keys given as argument.
    Args:
        dicts: list of dictionaries
        keys: keys to consider
        progress_bar: print a progress bar (default is False)
    Returns:
        Counter
    Example:
        >>> count_values([{'B-V': 'increase', 'B-ARGM-NEG': True},{'B-V': 'decrease'},{'B-V': 'decrease'}],keys = ['B-V'])
        Counter({'decrease': 2, 'increase': 1})
        >>> count_values([{'B-V': 'increase', 'B-ARGM-NEG': True},{'B-V': 'decrease'},{'B-V': 'decrease'}])
        Counter()
    """

    counts: Dict[str, int] = {}

    if progress_bar:
        print("Computing role frequencies...")
        time.sleep(1)
        dicts = tqdm(dicts)

    if keys is None:
        return Counter()

    for el in dicts:
        for key, value in el.items():
            if key in keys:
                if value in counts:
                    counts[value] += 1
                else:
                    counts[value] = 1

    return Counter(counts)



def count_words(sentences: List[str]) -> Counter:

    """
    A function that computes word frequencies in a list of sentences.
    Args:
        sentences: list of sentences
    Returns:
        Counter {"word": frequency}
    Example:
    >>> count_words(["this is a house"])
    Counter({'this': 1, 'is': 1, 'a': 1, 'house': 1})
    >>> count_words(["this is a house", "this is a house"])
    Counter({'this': 2, 'is': 2, 'a': 2, 'house': 2})
    >>> count_words([])
    Counter()
    """

    words: List[str] = []

    for sentence in sentences:
        words.extend(sentence.split())

    words_counter = Counter(words)

    return (words_counter)




def _get_wordnet_pos(word):
    """Get POS tag"""
    tag = pos_tag([word])[0][1][0].upper()

    return tag


wnl = WordNetLemmatizer()
f_lemmatize = wnl.lemmatize


def clean_text(
    sentences: List[str],
    remove_punctuation: bool = True,
    remove_digits: bool = True,
    remove_chars: str = "",
    stop_words: Optional[List[str]] = None,
    lowercase: bool = True,
    strip: bool = True,
    remove_whitespaces: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    tags_to_keep: Optional[List[str]] = None,
    remove_n_letter_words: Optional[int] = None,
) -> List[str]:

    """
    Clean a list of sentences.
    Args:
        sentences: list of sentences
        remove_punctuation: whether to remove string.punctuation
        remove_digits: whether to remove string.digits
        remove_chars: remove the given characters
        stop_words: list of stopwords to remove
        lowercase: whether to lower the case
        strip: whether to strip
        remove_whitespaces: whether to remove superfluous whitespaceing by " ".join(str.split(())
        lemmatize: whether to lemmatize using nltk.WordNetLemmatizer
        stem: whether to stem using nltk.SnowballStemmer("english")
        tags_to_keep: list of grammatical tags to keep (common tags: ['V', 'N', 'J'])
        remove_n_letter_words: drop words lesser or equal to n letters (default is None)
    Returns:
        Processed list of sentences
    Examples:
        >>> clean_text([' Return the factorial of n, an  exact integer >= 0.'])
        ['return the factorial of n an exact integer']
        >>> clean_text(['Learning is usefull.'])
        ['learning is usefull']
        >>> clean_text([' Return the factorial of n, an  exact integer >= 0.'], stop_words=['factorial'])
        ['return the of n an exact integer']
        >>> clean_text([' Return the factorial of n, an  exact integer >= 0.'], lemmatize=True)
        ['return the factorial of n an exact integer']
        >>> clean_text(['Learning is usefull.'],lemmatize=True)
        ['learn be usefull']
        >>> clean_text([' Return the factorial of n, an  exact integer >= 0.'], stem=True)
        ['return the factori of n an exact integ']
        >>> clean_text(['Learning is usefull.'],stem=True)
        ['learn is useful']
        >>> clean_text(['A1b c\\n\\nde \\t fg\\rkl\\r\\n m+n'])
        ['ab c de fg kl mn']
        >>> clean_text(['This is a sentence with verbs and nice adjectives.'], tags_to_keep = ['V', 'J'])
        ['is nice']
        >>> clean_text(['This is a sentence with one and two letter words.'], remove_n_letter_words = 2)
        ['this sentence with one and two letter words']
    """

    if lemmatize is True and stem is True:
        raise ValueError("lemmatize and stemming cannot be both True")

    #if stop_words is not None and lowercase is False:
    #    raise ValueError("remove stop words make sense only for lowercase")

    # remove chars
    if remove_punctuation:
        remove_chars += string.punctuation
    if remove_digits:
        remove_chars += string.digits
    if remove_chars:
        sentences = [re.sub(f"[{remove_chars}]", "", str(sent)) for sent in sentences]

    # lowercase, strip and remove superfluous white spaces
    if lowercase:
        sentences = [sent.lower() for sent in sentences]
    if strip:
        sentences = [sent.strip() for sent in sentences]
    if remove_whitespaces:
        sentences = [" ".join(sent.split()) for sent in sentences]

    # lemmatize
    if lemmatize:

        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }

        sentences = [
            " ".join(
                [
                    f_lemmatize(
                        word, tag_dict.get(_get_wordnet_pos(word), wordnet.NOUN)
                    )
                    for word in sent.split()
                ]
            )
            for sent in sentences
        ]

    # keep specific nltk tags
    # this step should be performed before stemming, but may be performed after lemmatization
    if tags_to_keep is not None:
        sentences = [
            " ".join(
                [
                    word
                    for word in sent.split()
                    if _get_wordnet_pos(word) in tags_to_keep
                ]
            )
            for sent in sentences
        ]

    # stem
    if stem:
        stemmer = SnowballStemmer("german")
        f_stem = stemmer.stem

        sentences = [
            " ".join([f_stem(word) for word in sent.split()]) for sent in sentences
        ]

    # drop stopwords
    # stopwords are dropped after the bulk of preprocessing steps, so they should also be preprocessed with the same standards
    if stop_words is not None:
        sentences = [
            " ".join([word for word in sent.split() if word not in stop_words])
            for sent in sentences
        ]

    # remove short words < n
    if remove_n_letter_words is not None:
        sentences = [
            " ".join(
                [word for word in sent.split() if len(word) > remove_n_letter_words]
            )
            for sent in sentences
        ]

    return (sentences)



def parsing(sentence):
    sent = nlp(sentence)

    results = []
    for word in sent:
        word_list = [word, word.dep_]
        res = [word, word_list]
        results.append(res)

    return(results)


def has_syn(ll, syn = "ROOT"):
    return(syn in str(ll))
"""
        elif word.dep_ == 'cj' and word.head == subj:
            result.append(word)
"""

def get_subjects(sent):
    sent = nlp(sent)
    result = []
    subj = None
    for word in sent:
        if 'sb' in word.dep_:
            subj = word
            result.append(word)
    return(result)



def get_objects(sent):
    sent = nlp(sent)
    result = []
    subj = None
    for word in sent:
        if 'da' in word.dep_:
            subj = word
            result.append(word)
        elif 'oa' in word.dep_:
                subj = word
                result.append(word)
        elif 'go' in word.dep_:
                    subj = word
                    result.append(word)
    return(result)


def get_rid(t):
    res = re.sub("(\, 'LOC'\)|, 'ORG'\)|, 'PER'\)|, 'MISC'\))", "", t)
    res = re.sub("\(", "", res)
    return(res)


def create_narr(sent):

    sent = nlp(sent)
    nar = []
    for word in sent:
        if "sb" in word.dep_:
            nar.append(word)
    for word in sent:
        if "ng" in word.dep_:
                nar.append(word)
    for word in sent:
        if "ROOT" in word.dep_:
                    nar.append(word)
    for word in sent:
        if 'da' in word.dep_:
            subj = word
            nar.append(word)
            next
        elif 'oa' in word.dep_:
            subj = word
            nar.append(word)
            next
    return(nar)


class SIF_word2vec:

    """
    A class to call a trained word2vec model using gensim's library.
    For basic code snippets and additional details: https://radimrehurek.com/gensim/models/word2vec.html
    """

    def __init__(
        self,
        path: str,
        sentences=List[str],
        alpha: Optional[float] = 0.001,
        normalize: bool = True,
    ):

        self._model = self._load_keyed_vectors(path)

        self._words_counter = count_words(sentences)

        self._sif_dict = compute_sif_weights(self._words_counter, alpha)

        self._vocab = self._model.vocab

        self._normalize = normalize

    def _load_keyed_vectors(self, path):
        return Word2Vec.load(path).wv

    def __call__(self, tokens: List[str]):
        res = np.mean(
            [self._sif_dict[token] * self._model[token] for token in tokens], axis=0
        )
        if self._normalize:
            res = res / norm(res)
        return res

    def most_similar(self, v):
        return self._model.most_similar(positive=[v], topn=1)[0]


class SIF_keyed_vectors(SIF_word2vec):

    """
    A class to call a pre-trained embeddings model from gensim's library.
    The embeddings are weighted by the smoothed inverse frequency of each token.
    For further details, see: https://github.com/PrincetonML/SIF
    # The list of pre-trained embeddings may be browsed by typing:
        import gensim.downloader as api
        list(api.info()['models'].keys())
    """


    def _load_keyed_vectors():
        vecs = KeyedVectors.load_word2vec_format("vec_glove.txt", binary=False)
        return (vecs)




def get_vector(tokens: List[str], model: SIF_keyed_vectors):

    """
    A function that computes an embedding vector for a list of tokens.
    Args:
        tokens: list of string tokens to embed
        model: trained embedding model. It can be either:
         - Universal Sentence Encoders (USE)
         - a full gensim Word2Vec model (SIF_word2vec)
         - gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)
    Returns:
        A two-dimensional numpy array (1,dimension of the embedding space)


    if not isinstance(model, (USE, SIF_word2vec, SIF_keyed_vectors)):
        raise TypeError("Union[USE, SIF_Word2Vec, SIF_keyed_vectors]")
        """

    if isinstance(model, SIF_word2vec) or isinstance(model, SIF_keyed_vectors):
        if not tokens:
            res = None
        elif any(token not in model._sif_dict for token in tokens):
            res = None
        elif any(token not in model._vocab for token in tokens):
            res = None
        else:
            res = model(tokens)
            res = np.array(
                [res]
            )  # correct format to feed the vectors to sklearn clustering methods
    else:
        res = model(tokens)
        res = np.array(
            [res]
        )  # correct format to feed the vectors to sklearn clustering methods

    return res



def compute_sif_weights(words_counter: dict, alpha: Optional[float] = 0.001) -> dict:

    """
    A function that computes smooth inverse frequency (SIF) weights based on word frequencies.
    (See "Arora, S., Liang, Y., & Ma, T. (2016). A simple but tough-to-beat baseline for sentence embeddings.")
    Args:
        words_counter: a dictionary {"word": frequency}
        alpha: regularization parameter
    Returns:
        A dictionary {"word": SIF weight}
    """

    sif_dict = {}

    for word, count in words_counter.items():
        sif_dict[word] = alpha / (alpha + count)

    return sif_dict

from nltk.tokenize import word_tokenize
def del_stopwords(sent, stopwords, method = "nltk"):
    # spacy method could also be implemtented for comparison
    if method == "nltk":
        # tokenize
        toks = word_tokenize(sent)
        # remove the words not in stopwords
        res = [tok for tok in toks if tok.lower() not in stopwords]
    return " ".join(res)

def del_stopwords_list(test, sp_stopwords):
    t = test
    to_del = []
    for i in range(len(t)):
        if str(t[i][0]) in sp_stopwords:
            #print("delete " + str(t[i][0]))
            to_del.append(i)
    for index in sorted(to_del, reverse=True):
        del t[index]
    return(t)


def del_stopwords_list_adj(test, sp_stopwords):
    t = test
    to_del = []
    for i in range(len(t)):
        if str(t[i]) in sp_stopwords:
            #print("delete " + str(t[i]))
            to_del.append(i)
    for index in sorted(to_del, reverse=True):
        del t[index]
    return(" ".join(str(element) for element in t))



def del_punct_list(test):
    t = test
    to_del = []
    for i in range(len(t)):
        if str(t[i][1][1]) == 'punct':
            #print("delete " + str(t[i]))
            to_del.append(i)
    for index in sorted(to_del, reverse=True):
        del t[index]
    return(t)



def del_number_list(test):
    t = test
    to_del = []
    for i in range(len(t)):
        if str(t[i][0]).isnumeric():
            #print("delete " + str(t[i]))
            to_del.append(i)
    for index in sorted(to_del, reverse=True):
        del t[index]
    return(t)


def del_number_list_adj(test):
    t = test
    to_del = []
    for i in range(len(t)):
        if str(t[i]).isnumeric():
            #print("delete " + str(t[i]))
            to_del.append(i)
    for index in sorted(to_del, reverse=True):
        del t[index]
    return(" ".join(str(element) for element in t))



def lemma_list(test):
    sent = []

    for i in range(len(test)):
        sent.append(str(test[i][0]))

    t = " ".join(sent)

    doc = nlp(t)

    for t in doc:
        result = [x.lemma_ for x in doc]

    return(result)


def get_narr_elements_list(text):

    # potential filter for short texts
    #if len(text) < 8:
    #    return(text)

    # then look at triplets in text list structure
    dat = []

    for i in range(len(text)):
        dat.append(text[i][1])

    sentence = []

    for i in range(len(dat)):
        sentence.append(str(dat[i][0]))

    # flatten list
    flatten_list = []

    for subl in dat:
        for item in subl:
            flatten_list.append(item)

    # nr of subjects
    nr_sb = str(flatten_list).count("sb")

    # nr roots
    nr_sb = str(flatten_list).count("ROOT")

    # if no subject, return empty list
    if nr_sb == 0:
        print("no subject found, return empty list")
        return([])


    # get the quasi-sentence with ROOT clause
    for i in range(len(dat)):
        if dat[i][1] == "ROOT":
            verb = dat[i][0]
            root_child = []
            for a in dat[i][3]:
                print(a.text)
                if str(a.text) in str(sentence):
                    root_child.append(a)
                #print(root_child)
            return([verb, root_child])
        else:
            next

    return(0)


def get_root_sent(sent_list):
    if "ROOT" not in str(sent_list):
        return([])
    else:
        root_l = [l[1][0] for l in sent_list if 'ROOT' in str(l[1])][0]
        res = [i for i in root_l.children if str(i.dep_) in ["sb", "sbp"]]
        res.append(root_l)
        add = [i for i in root_l.children if str(i.dep_) in ["oa", "ng", "da", "nk", "og", "oc"]]
        res = res + add

        return(res)


def get_sb_sent(sent_list, subj_tags = ["sb", "sbp"]):
    if any(subj_tags) not in str(sent_list):
        return([])
    else:
        sb_l = [l[1][0] for l in sent_list if any(subj_tags) in str(l[1])]
        res = []
        print(sb_l)
        if len(sb_l) < 2:
            res.append(sb_l)
            res = [i for i in sb_l[0].ancestors if str(i.dep_) in ["ROOT", "svp"]]
            res = res + [i for i in sb_l[0].ancestors if str(i.dep_) in ["oa", "ng", "da", "nk"]]
            return(res)
        else:
            j = 1
            while len(sb_l) > j:
                res.append(sb_l[j-1])
                res = [i for i in sb_l[j-1].ancestors if str(i.dep_) in ["ROOT", "svp"]]
                res = res + [i for i in sb_l[j-1].ancestors if str(i.dep_) in ["oa", "ng", "da", "nk"]]
                return(res)





def is_negation(tok):
    negs = [i for i in tok.children if str(i.dep_) == "ng"] # using the syntactic parsing output ("ng" tag for German model)
    return (negs)

# check whether negation is in list of verbs
def is_negation_l(v_list):
    negs = []
    for tok in v_list:
        neg_tok = [i for i in tok.children if str(i.dep_) == "ng"]
        if len(neg_tok) > 0:
            negs.append(neg_tok[0])
    return (negs)


def filter_pos(sent, pos = ["VERB", "AUX"], dep = "ROOT"):
    """
    Returns all tokens with specific part of speech tags.
    """
    l = [tok for tok in sent if tok.pos_ in pos or tok.dep_ in dep]
    return l



def get_verb(sent):
    """
    Returns verb tokens.
    """
    verbs = []

    for tok in sent:

        # case AUX
        if  tok.pos_ == "AUX":
            desc = [d for d in tok.children]
            #print([d.pos_ for d in desc])
            if "AUX" in [d.pos_ for d in desc] or "VERB" in [d.pos_ for d in desc]:
                # TO IMPLEMENT: Go further up the syntactic tree for triplet "wird gegessen haben"
                r = [tok]
                ch = [t for t in tok.children if t.pos_ in ["AUX", "VERB"]]
                r.append(ch[0])
                verbs.append(r)

        # case VERB
        elif tok.pos_ == "VERB":
            # When AUX exists, don't include because case AUX
            anc = tok.ancestors
            if "AUX" not in [a.pos_ for a in anc]:
                r = [tok]
                #r.append(t.token for t in tok.subtree if t.pos_ == "AUX")
                verbs.append(r)

    return verbs




def get_deps(verb, deps=None):
    """
    Returns all dependencies of a verb.
    """
    l = []
    if deps is not None:
        l.extend([tok for tok in verb.lefts if tok.dep_ in deps])
        l.extend([tok for tok in verb.rights if tok.dep_ in deps])
    else:
        l.extend([tok for tok in verb.lefts])
        l.extend([tok for tok in verb.rights])
    return l


def get_text(tokens):
    """
    Returns text from list of spacy tokens.
    """
    return [tok.text for tok in tokens]





def sentence_processing(split_sentences, stopwords, lowercase = True):

    """
    function that takes in split sentences and performs the whole shabang until
    clustering and ner.
    """
    res_l = []

    for i in range(len(split_sentences[0])):
        # get doc_id, sentence id for each sentence
        doc_id = split_sentences[0][i]
        sent_id = i
        full_sent = split_sentences[1][i]
        # run through nlp model
        sent = parsing(split_sentences[1][i])

        # delete stopwords
        sent = del_stopwords_list(sent, stopwords)

        # delete punctuation
        sent = del_punct_list(sent)

        # delete numeric
        sent = del_number_list(sent)

        """
        # get list of lemmatized words
        lemmas = lemma_list(sent)

        print(sent)
        # change to lemmatized when suitable
        for i in range(len(sent)):
            if sent[i][1][1] == "sb":
                sent[i][0] = str(sent[i][0]).lower()
                next
            else:
                sent[i][0] = lemmas[i].lower()
        """


        # get relevant entities
        res = get_root_sent(sent)

        r = [doc_id, sent_id, full_sent, res]

        #print(r)
        if len(res) < 3 or len(res) > 6:
            next
        else:
            res_l.append(r)

        if i % 100 == 50:
            print(str(i) + " of " + str(len(split_sentences[1])))

    return(res_l)




def full_narr_extraction(split_sentences, stopwords, lowercase = True):

    res_nars = []

    for i in range(len(split_sentences[0])):
        # get doc_id, sentence id for each sentence
        doc_id = split_sentences[0][i]
        sent_id = i
        full_sent = split_sentences[1][i]

        # run through nlp model
        sent = nlp(split_sentences[1][i])

        # sent = parsing(split_sentences[1][i])

        # all_verbs = filter_pos(sent, pos=["VERB", "AUX"])
        all_verbs = get_verb(sent)

        statements = []
        for j, verb in enumerate(all_verbs):

            negation = is_negation_l(verb)

            # subjects
            subjs = []
            subjs.extend(get_deps(verb[0], deps=["sb"]))  # active forms
            subjs.extend(get_deps(verb[0], deps=["sbp"]))  # passive forms
            if len(verb) > 1:
                subjs.extend(get_deps(verb[1], deps=["sb", "sbp"]))  # active forms

            """
            for k, subj in enumerate(subjs):
                if subj.text in ["qui", "qu'"]:
                    for tok in sent:
                        for t in tok.rights:
                            if t == verb:
                                subjs[k] = tok
                        for t in tok.lefts:
                            if t == verb:
                                subjs[k] = tok
            """

            if len(subjs) != 0:
                subjs = [" ".join([t.text for t in subj.subtree]) for subj in subjs]

            elif j > 0 and len(statements) > 0:
                subjs = [statements[j - 1][1]]

            #print("subs: ", subjs)

            # objects
            objs = []
            objs.extend(get_deps(verb[0], deps=["oa", "og", "da", "op", "pg"]))  # active forms
            if len(verb) > 1:
                objs.extend(get_deps(verb[1], deps=["oa", "og", "da", "op", "pg"]))  # active forms
            # objs.extend(get_deps(verb, deps=["op"]))  # passive forms

            """
            for k, obj in enumerate(objs):
                if obj.text in ["que", "qu'"]:
                    for tok in sent:
                        for t in tok.rights:
                            if t == verb:
                                objs[k] = tok
                        for t in tok.lefts:
                            if t == verb:
                                objs[k] = tok
            """

            if len(objs) != 0:
                objs = [" ".join([t.text for t in obj.subtree]) for obj in objs]

            # packaging
            subjs = " ".join(subjs)
            objs = " ".join(objs)
            #verb = verb[0].text

            svo = (j, subjs, negation, verb, objs)

            statements.append(svo)

            """
            # delete stopwords
            sent = del_stopwords_list(sent, stopwords)

            # delete punctuation
            sent = del_punct_list(sent)

            # delete numeric
            sent = del_number_list(sent)
            """

        res_nars.append([doc_id, sent_id, sent, statements])

        if i % 100 == 50:
            print(str(i) + " of " + str(len(split_sentences[1])))

    return res_nars





def get_entities(word_list):
    print(word_list[2])
    sent = " ".join(word_list[2])
    doc = nlp(sent)
    ents = [ent for ent in doc.ents]
    return(len(ents))


### utils for clustering from relatio


"""
def get_vector(tokens: List[str], model: Union[USE, SIF_word2vec, SIF_keyed_vectors]):
"""
"""
A function that computes an embedding vector for a list of tokens.
Args:
    tokens: list of string tokens to embed
    model: trained embedding model. It can be either:
     - Universal Sentence Encoders (USE)
     - a full gensim Word2Vec model (SIF_word2vec)
     - gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)
Returns:
    A two-dimensional numpy array (1,dimension of the embedding space)
"""
"""
    if not isinstance(model, (USE, SIF_word2vec, SIF_keyed_vectors)):
        raise TypeError("Union[USE, SIF_Word2Vec, SIF_keyed_vectors]")

    if isinstance(model, SIF_word2vec) or isinstance(model, SIF_keyed_vectors):
        if not tokens:
            res = None
        elif any(token not in model._sif_dict for token in tokens):
            res = None
        elif any(token not in model._vocab for token in tokens):
            res = None
        else:
            res = model(tokens)
            res = np.array(
                [res]
            )  # correct format to feed the vectors to sklearn clustering methods
    else:
        res = model(tokens)
        res = np.array(
            [res]
        )  # correct format to feed the vectors to sklearn clustering methods

    return res
"""

def get_vector(tokens, model):
    res = model(tokens)
    res = np.array(
        [res]
    )  # correct format to feed the vectors to sklearn clustering methods

    return res

def get_vectors(
    postproc_roles,
    #model: Union[USE, SIF_word2vec, SIF_keyed_vectors],
    model,
    used_roles=List[str],
):

    """
    A function to train a K-Means model on the corpus.
    Args:
        postproc_roles: list of statements
        model: trained embedding model. It can be either:
         - Universal Sentence Encoders (USE)
         - a full gensim Word2Vec model (SIF_word2vec)
         - gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)
        used_roles: list of semantic roles to cluster together
    Returns:
        A list of vectors
    """

    role_counts = count_values(postproc_roles, keys=used_roles)

    role_counts = [role.split() for role in list(role_counts)]

    vecs = []
    for role in role_counts:
        vec = get_vector(role, model)
        if vec is not None:
            vecs.append(vec)

    vecs = np.concatenate(vecs)

    return vecs





def train_cluster_model(
    vecs,
    #model: Union[USE, SIF_word2vec, SIF_keyed_vectors],
    n_clusters,
    random_state: Optional[int] = 0,
    verbose: Optional[int] = 0,
):

    """
    Train a kmeans model on the corpus.
    Args:
        vecs: list of vectors
        model: trained embedding model. It can be either:
         - Universal Sentence Encoders (USE)
         - a full gensim Word2Vec model (SIF_word2vec)
         - gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)
        random_state: seed for replication (default is 0)
        verbose: see Scikit-learn documentation for details
    Returns:
        A Scikit-learn kmeans model
    """

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, verbose=verbose
    ).fit(vecs)

    return kmeans




def get_clusters(
    postproc_roles: List[dict],
    model,
    kmeans,
    used_roles=List[str],
    progress_bar: bool = False,
    suffix: str = "_lowdim",
) -> List[dict]:

    """
    Predict clusters based on a pre-trained kmeans model.
    Args:
        postproc_roles: list of statements
        model: trained embedding model
        (e.g. either Universal Sentence Encoders, a full gensim Word2Vec model or gensim Keyed Vectors)
        kmeans = a pre-trained sklearn kmeans model
        used_roles: list of semantic roles to consider
        progress_bar: print a progress bar (default is False)
        suffix: suffix for the new dimension-reduced role's name (e.g. 'ARGO_lowdim')
    Returns:
        A list of dictionaries with the predicted cluster for each role
    """

    roles_copy = deepcopy(postproc_roles)

    if progress_bar:
        print("Assigning clusters to roles...")
        time.sleep(1)
        postproc_roles = tqdm(postproc_roles)

    for i, statement in enumerate(postproc_roles):
        for role, tokens in statement.items():
            if role in used_roles:
                vec = get_vector(tokens.split(), model)
                if vec is not None:
                    clu = kmeans.predict(vec)[0]
                    roles_copy[i][role] = clu
                else:
                    roles_copy[i].pop(role, None)
            else:
                roles_copy[i].pop(role, None)

    roles_copy = [
        {str(k + suffix): v for k, v in statement.items()} for statement in roles_copy
    ]

    return roles_copy





def label_clusters_most_freq(
    clustering_res: List[dict], postproc_roles: List[dict]
) -> dict:

    """
    A function which labels clusters by their most frequent term.
    Args:
        clustering_res: list of dictionaries with the predicted cluster for each role
        postproc_roles: list of statements
    Returns:
        A dictionary associating to each cluster number a label (e.g. the most frequent term in this cluster)
    """

    temp = {}
    labels = {}

    for i, statement in enumerate(clustering_res):
        for role, cluster in statement.items():
            tokens = postproc_roles[i][role]
            cluster_num = cluster
            if cluster_num not in temp:
                temp[cluster_num] = [tokens]
            else:
                temp[cluster_num].append(tokens)

    for cluster_num, tokens in temp.items():
        token_most_common = Counter(tokens).most_common(2)
        if len(token_most_common) > 1 and (
            token_most_common[0][1] == token_most_common[1][1]
        ):
            warnings.warn(
                f"Multiple labels for cluster {cluster_num}- 2 shown: {token_most_common}. First one is picked.",
                RuntimeWarning,
            )
        labels[cluster_num] = token_most_common[0][0]

    return labels





def label_clusters_most_similar(kmeans, model) -> dict:

    """
    A function which labels clusters by the term closest to the centroid in the embedding
    (i.e. distance is cosine similarity)
    Args:
        kmeans: the trained kmeans model
        model: trained embedding model. It can be either:
         - a full gensim Word2Vec model (SIF_word2vec)
         - gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)
    Returns:
        A dictionary associating to each cluster number a label
        (e.g. the most similar term to cluster's centroid)
    """

    labels = {}

    for i, vec in enumerate(kmeans.cluster_centers_):
        most_similar_term = model.wv.most_similar(vec)
        labels[i] = most_similar_term[0]

    return labels
