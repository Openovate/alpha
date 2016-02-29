# -->

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

from alpha.lib import tokenizers
from alpha.lib import analyzers

from itertools import product

import numpy as np
import collections
import utils
import re

# data processing for training
# and data clustering
class Processor(object):
    # continues features
    CONTINUOUS_FEATURES = {
        'width': lambda page, datapoint: float(datapoint['bound']['width']),
    }

    # data initialization
    def __init__(
        self,
        data,
        tokenizer=tokenizers.EnglishTokenizer,
        analyzer=analyzers.TermFrequencyAnalyzer
    ):
        # load up the given data
        self.data       = data
        # initialize tokenizer
        self.tokenizer  = tokenizer()

        # tokenize data fields in each page
        pages = []
        # iterate on each pages
        for page in self.data:
            # tokenize titles
            page['titles']       = self.tokenizer.tokenize(*page['titles'])
            # tokenize descriptions
            page['descriptions'] = self.tokenizer.tokenize(*page['descriptions'])
            # merge tokenized titles and descriptions
            tokens = page['titles'] + page['descriptions']

            # iterate on each page text
            for text in page['texts']:
                # set default label
                text['label']  = 0
                # tokenize text
                text['tokens'] = self.tokenizer.tokenize(*text['text'])
                # stack up tokenized text
                tokens += text['tokens']

            # append tokens
            pages.append(tokens)

        # initialize our analyzer
        self.analyzer = analyzer(*pages)

        # pages track
        self.pages = []
        # texts tract
        self.texts = []

        # iterate on each page
        for page in self.data:
            # iterate on each page texts
            for text in page['texts']:
                # keep track of corresponding page and text for each datapoint
                self.pages.append(page)
                # keep track of page texts
                self.texts.append(text)

    # data extration / feature selection
    def extract(self):
        # continues features (non fixed features)
        continuous_features = []
        # discrete features (fixed features)
        discrete_features   = []

        # iterate on each page and texts
        for page, text in zip(self.pages, self.texts):
            # continuous features
            continuous_features.append([
                process(page, text)
                for key, process in self.CONTINUOUS_FEATURES.iteritems()
            ])

            # discrete features for computed styles
            discrete_feature = dict(text['computed'].items())
            # descrete features for tag path
            discrete_feature['path'] = ' > '.join(text['path'])
            # append discrete features for the given page text
            discrete_features.append(discrete_feature)

        # build numpy array
        continuous_features = preprocessing.scale(np.array(continuous_features))

        # vectorize discrete features
        vectorizer        = DictVectorizer()
        # transform discrete features into vectorized data
        discrete_features = vectorizer.fit_transform(discrete_features).toarray()

        # stack up continuous and discrete features
        return np.hstack([continuous_features, discrete_features]).astype(np.float32)

    # prepare data for svm
    def prepare(self, labels):
        # clusters structure
        clusters = collections.defaultdict(lambda: dict(
            label=0,
            score=0.0,
            pages=collections.defaultdict(lambda: dict(
                texts=[],
            )),
        ))

        # iterate over each block
        for page, text, label in zip(self.pages, self.texts, labels):
            # first find out this text block's relevence score
            hints = page['titles'] + page['descriptions']
            # get similarity between meta text and meta descriptions
            score = self.analyzer.get_similarity(text['tokens'], hints) if hints else 0.0

            # find the cluster
            cluster           = clusters[int(label)]
            # increment score
            cluster['score'] += score

            # append texts
            cluster['pages'][page['url']]['texts'].append(text)

        #  compute best cluster
        best_cluster = max(clusters.values(), key=lambda x: x['score'])
        # iterate on each best cluster
        for page in best_cluster['pages'].values():
            # iterate on each text and label it
            for text in page['texts']:
                text['label'] = 1

        # build continuous features
        continuous_features = []
        # build discrete features
        discrete_features   = []
        # build labels
        labels              = []

        # iterate on each texts
        for text in self.texts:
            # get text length
            text_length  = len(text['tokens'])
            # get bounding area
            area         = text['bound']['height'] * text['bound']['width']
            # compute text density
            text_density = float(text_length) / float(area)

            # continuous_feature
            continuous_feature = [text_length, text_density, float(text_length) / float(len(text['text'])), area]
            # append text feature
            continuous_features.append(continuous_feature)

            # discrete features
            discrete_feature = dict()
            
            # set selectors as discrete feature
            discrete_feature['class'] = ' > '.join([
                '%s%s' % (
                    selector['name'],
                    '.' + '.'.join(selector['classes']) if selector['classes'] else '',
                )
                for selector in text['selector']
            ])
            
            # append discrete feature
            discrete_features.append(discrete_feature)

            # append label
            labels.append(text['label'])

        return continuous_features, discrete_features, labels
    
    # score clusters
    def score(self, labels):
        # clusters structure
        clusters = collections.defaultdict(lambda: dict(
            score=0.0,
            selectors=[],
            pages=collections.defaultdict(lambda: dict(
                score=0.0,
                tokens=[],
                text=[],
            )),
        ))

        # iterate on each page
        for page, text, label in zip(self.pages, self.texts, labels):
            # combine meta titles and meta descriptions as hints
            hints = page['titles'] + page['descriptions']
            # compute relevance score
            relevance_score = self.analyzer.get_similarity(text['tokens'], hints) if hints else 1.0
            
            # get cluster
            cluster = clusters[int(label)]
            # append text selector to cluster
            cluster['selectors'].append(text['selector'])
            # increment relevance score
            cluster['pages'][page['url']]['score'] += relevance_score
            # append tokens
            cluster['pages'][page['url']]['tokens'].append(text['tokens'])
            # append text
            cluster['pages'][page['url']]['text'] += text['text']

        # iterate on each cluster values
        for cluster in clusters.values():
            # count non zero pages
            count = 0

            # coherence score
            for page in cluster['pages'].values():
                # compute coherent score
                coherent_score = 0.0
                # iterate on each tokens
                for tokens1, tokens2 in product(page['tokens'], repeat=2):
                    # if tokens does not match
                    # we are calculating lcs here
                    if tokens1 is not tokens2:
                        # increment coherent score
                        coherent_score += self.analyzer.get_similarity(tokens1, tokens2)

                # do we have enough tokens?
                if len(page['tokens']) <= 1:
                    # set default coherent score
                    coherent_score = 1.0

                # delete page tokens
                del page['tokens']

                # combine scores
                page['score']    *= coherent_score
                # combine cluster score in page score
                cluster['score'] += page['score']

                # increment page score
                if page['score'] > 0:
                    count += 1

                # normalize content
                page['content'] = re.sub(r'[^a-zA-Z0-9]+', ' ', ' '.join(page['text']))
                page['content'] = re.sub(r'[\s]{2,}', ' ', ' '.join(page['text'])).strip()

            # compute cluster score
            if count > 0: cluster['score'] /= float(count)

            # calculate confidence score
            cluster['confidence'] = float(count) / float(len(cluster['pages']))

            # consolidate clusters
            cluster['selectors'] = utils.consolidate_selectors(cluster['selectors'])

        # get rid of the clusters with score 0
        for label in clusters.keys():
            # skip zero score or zero confidence
            if clusters[label]['confidence'] <= 0 or clusters[label]['score'] <= 0:
                # delete it on our cluster
                del clusters[label]

        return clusters.values()