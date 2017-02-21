"""
    Train a logistic regresion model for document classification.

    Search this file for the keyword "Hint" for possible areas of
    improvement.  There are of course others.
"""
#from distutils.version import LooseVersion as Version
#from sklearn import __version__ as sklearn_version
import pandas as pd
import pickle
#import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB

# Used for porterStemmer tokenizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

"""
    This is a very basic tokenization strategy.

"""
porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]



# Read in the dataset and store in a pandas dataframe
print("Reading Data")
df = pd.read_csv('./training_movie_data.csv')

   
# Perform feature extraction on the text.
print("Start preprocessor")
tfidf = TfidfVectorizer(strip_accents=None, lowercase=True, preprocessor=None,
                        tokenizer = tokenizer_porter,
                        sublinear_tf=True, stop_words='english', max_df = .7,
                        min_df = 0)


# Create a pipeline to vectorize the data and then perform regression.
# Uses Multinomial Naive Bayes classifier.
print("Creating pipeline")
lr_tfidf = Pipeline([
                    ('vect', tfidf),
                    ('clf', MultinomialNB())
                    ])


"""
Perform K-fold validation with 10 folds
"""
from sklearn.cross_validation import KFold
print("Performing K-Fold validation with 10 folds")
k_fold = KFold(n=len(df), n_folds=10)
scores = []
x = 0
for train_indices, test_indices in k_fold:
    print("ENTERED K-FOLD LOOP")
    
    train_x = df.iloc[train_indices]['review'].values
    train_y = df.iloc[train_indices]['sentiment'].values
                     
    test_x = df.iloc[test_indices]['review'].values
    test_y = df.iloc[test_indices]['sentiment'].values

    lr_tfidf.fit(train_x, train_y)

    score = lr_tfidf.score(test_x, test_y)
    scores.append(score)
    print(x)
    x = x+1

print('Total reviews classified:', len(df))
print('Score:', sum(scores)/len(scores))#this is taking the average score from the cross validation

# Save the classifier for use later.
pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
