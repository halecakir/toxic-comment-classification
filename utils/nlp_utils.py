import re
import string

import contractions
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")


class NLPUtils:
    @staticmethod
    def preprocess_pipeline(data, lemmatization=True, stemming=True):
        preprocessed_data = []
        for sent in data:
            # convert to lowercase
            sent = sent.lower()

            # remove hyperlinks
            sent = NLPUtils.hyperlinks_remove(sent)

            # remove non-ascii chars
            sent = NLPUtils.non_ascii_remove(sent)

            # remove digits
            sent = NLPUtils.digits_remove(sent)

            # remove puntuations
            sent = NLPUtils.punctuation_remove(sent)

            # expand contractions
            sent = contractions.fix(sent)

            # tokenize words
            words = word_tokenize(sent)

            # eliminate stopwords111
            words = NLPUtils.stopword_elimination(words)

            # lemmatize words
            if lemmatization:
                words = NLPUtils.lemmatize(words)

            # stem words
            if stemming:
                words = NLPUtils.stem(words)

            preprocessed_data.append(words)
        return preprocessed_data

    @staticmethod
    def stopword_elimination(words):
        return [word for word in words if word not in stopwords.words("english")]

    @staticmethod
    def hyperlinks_remove(text):
        regex = r"((https?:\/\/)?[^\s]+\.[^\s]+)"
        text = re.sub(regex, "", text)
        return text

    @staticmethod
    def punctuation_remove(text):
        d = {w: " " for w in string.punctuation}
        translator = str.maketrans(d)
        return text.translate(translator)

    @staticmethod
    def lemmatize(words):
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word) for word in words]
        return lemmas

    @staticmethod
    def stem(words):
        stemmer = nltk.SnowballStemmer("english")
        stems = [stemmer.stem(word) for word in words]
        return stems

    @staticmethod
    def non_ascii_remove(text):
        return re.sub(r"[^\x00-\x7f]", r"", text)

    @staticmethod
    def digits_remove(text):
        text = re.sub(r"[0-9]+", "", text)
        return text
