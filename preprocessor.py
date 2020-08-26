import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


from helpers import Clock
from libherokuserver import *


class Preprocessor():
    """Class created to be able to pickle"""

    def __init__(self, frac=0.03, word_base='stem', top_words_count=1000, test_total_ratio=0.2):
        self.frac=frac
        self.word_base=word_base
        self.top_words_count=top_words_count
        self.test_total_ratio=test_total_ratio
        
    def run(self):
        """Run preprocessing"""
        
        clock = Clock(f'Preproc {self.frac:4}', notify_os=True)
        
        # load & clean data
        df = load_init(frac=self.frac)
        df['category'], labels = pd.factorize(df['category'])
        self.label_map = {v: i for i, v in list(enumerate(labels))}
        df['message'] = df['message'].apply(clear_message)

        df.dropna(inplace=True)

        clock.tik(f'Basing... ({self.word_base})')
        self.stop_set = get_sw([
            'english',
            'russian',
            'chinees'
        ])
        self.baser = get_baser(word_base)
    
    
        # base message words
        df['based_words'] = df['message'].apply(lambda x: basing(x, self.baser, self.stop_set))
        self.top_words = get_top_words(df['based_words'], top_N=self.top_words_count)
        
        
        # filter words
        filter_words = lambda words: [word for word in words if word in self.top_words]
        df['filtered_words'] = df['based_words'].apply(filter_words)

        clock.tik('Basing done', show_time=True)
        
        
        # vectorize
        vec_result = get_vectorized(df['filtered_words'].apply(lambda x: ' '.join(x)), 
                                    df['category'], 
                                    top_words_count=self.top_words_count)
        train, train_columns, target, self.vectorizer = vec_result

        
        # oversample
        clock.tik('Oversampling...')
        ros = RandomOverSampler()
        X_ros, y_ros = ros.fit_sample(train[train_columns], train[target])
        clock.tik('Oversampling done!', show_time=True)
        X_ros[target] = y_ros
        train = X_ros[:]

        
        # split on train and test
        x_train, x_test, y_train, y_test = train_test_split( train[train_columns]
                                                           , train[target] 
                                                           , test_size=self.test_total_ratio
                                                           , random_state=RANDOM_STATE 
                                                           , stratify = train[target] 
                                                           )
        clock.tik(f" Train: {train.shape}")
        return x_train, x_test, y_train, y_test
    

    def message_prepare(self, message):
        """Vectorize message"""
        based = basing(message, self.baser, self.stop_set)
        
        filter_words = lambda words: [word for word in words if word in self.top_words]
        filtered = filter_words(based)
        
        vectorized =  self.vectorizer.transform([' '.join(filtered)]).toarray()
        return vectorized
