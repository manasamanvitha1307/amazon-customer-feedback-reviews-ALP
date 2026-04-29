import string
from nltk.corpus import stopwords

def text_process(review):
    nopunc = ''.join([c for c in review if c not in string.punctuation])
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]