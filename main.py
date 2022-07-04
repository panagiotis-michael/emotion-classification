#!/usr/bin/env python

'''
Function that trains a specific model and based on train data
and gets predictions for test data and exports them in a text file, one per line.


__author__ = "Panagiotis Michael"
__copyright__ = "Copyright 2022"
__license__ = "MIT LICENSE"
__version__ = "0.1"
__maintainer__ = "Panagiotis Michael"
__email__ = "panagiotis.michael133@gmail.com"
__status__ = "Finished"

'''

'''
Please install the following dependencies:
1. pandas with pip install pandas
2. neattext with pip install neattext
3. nltk with pip install nltk
4. sklearn with pip install scikit-learn
'''

''' Note extra wordlists and tags are required to be install within the method. '''

''' train and test are the file names in string format. If they are not in the same directory pass the filenames with the directories. '''
def train_test(train,test):
    import pandas as pd

    import neattext.functions as nfx
    import nltk
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag

    from sklearn.pipeline import Pipeline

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsRestClassifier

    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    
    # Lemmatization is slower than stemming but it is generally more effective. It gives more context and meaning to words. It preserves them
    class LemmaTokenizer:
        def __init__(self):
            self.lemma = WordNetLemmatizer()
        def __call__(self, text):
            tokenized = word_tokenize(text)
            lemmatized=[]
        
            for token,tag in pos_tag(tokenized):
                pos=tag[0].lower()
        
                if pos not in ['a', 'r', 'n', 'v']:
                    pos='n'
            
                lemmatized.append(self.lemma.lemmatize(token,pos))    
        
            return lemmatized
    
    # Get data
    train = pd.read_csv(train)
    test =  pd.read_csv(test)
    
    test = test['text']
    
    # Preprocess
    train['text'] = train['text'].apply(lambda x: x.lower())
    train['text'] = train['text'].apply(nfx.remove_multiple_spaces)
    train['text'] = train['text'].apply(nfx.remove_punctuations)
    train['text'] = train['text'].apply(nfx.remove_puncts)
    train['text'] = train['text'].apply(nfx.remove_stopwords)
    train['text'] = train['text'].apply(nfx.remove_emojis)
    train['text'] = train['text'].apply(nfx.remove_special_characters)
    train['text'] = train['text'].apply(nfx.remove_bad_quotes)
    train['text'] = train['text'].apply(nfx.remove_non_ascii)
    train['text'] = train['text'].apply(nfx.remove_accents)
    train['text'] = train['text'].apply(nfx.remove_urls)
    train['text'] = train['text'].apply(nfx.remove_html_tags)
    train['text'] = train['text'].apply(nfx.remove_userhandles)
    train['text'] = train['text'].apply(nfx.remove_hashtags)
    train['text'] = train['text'].apply(nfx.remove_phone_numbers)

    test['text'] = test['text'].apply(lambda x: x.lower())
    test['text'] = test['text'].apply(nfx.remove_multiple_spaces)
    test['text'] = test['text'].apply(nfx.remove_punctuations)
    test['text'] = test['text'].apply(nfx.remove_puncts)
    test['text'] = test['text'].apply(nfx.remove_stopwords)
    test['text'] = test['text'].apply(nfx.remove_emojis)
    test['text'] = test['text'].apply(nfx.remove_special_characters)
    test['text'] = test['text'].apply(nfx.remove_bad_quotes)
    test['text'] = test['text'].apply(nfx.remove_non_ascii)
    test['text'] = test['text'].apply(nfx.remove_accents)
    test['text'] = test['text'].apply(nfx.remove_urls)
    test['text'] = test['text'].apply(nfx.remove_html_tags)
    test['text'] = test['text'].apply(nfx.remove_userhandles)
    test['text'] = test['text'].apply(nfx.remove_hashtags)
    test['text'] = test['text'].apply(nfx.remove_phone_numbers)
    
    # Pipeline
    sgd = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(),ngram_range=(1,2))),
    ('tfidf', TfidfTransformer()),
    ('sgd', OneVsRestClassifier(SGDClassifier(loss='hinge',alpha=0.0001,penalty='l1',learning_rate='optimal',class_weight=None)))
    ])
    
    # Fit and cross validate
    sgd.fit(train['text'],train['emotion'])
    
    # Get predictions from test dataset
    predictions = sgd.predict(test)
    
    # Write to predictions text file
    textfile = open("predictions.txt", "w")
    
    for element in predictions:
        textfile.write(element + "\n")
        
    textfile.close()
    
    return