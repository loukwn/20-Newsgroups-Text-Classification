import os
import string
from math import ceil, log
from random import randint

import nltk
from nltk import WordNetLemmatizer

from classifier.indexer import DocumentIndexer
from classifier.metrics import VectorMetrics


class DocClassifier:
    _category_dict = {}
    _document_database_path = '20news-19997/20_newsgroups/'
    _tagged_docs_path = 'data/tagged_docs/'
    _characteristics = {}
    _category_models = {}
    _characteristics_strings = []
    _total_docs_in_e = 0

    def __init__(self, categories=None, docs_per_cat=-1, training_ratio=0.5, characteristic_num=1000, silent=False,
                 metric_type=1):

        if categories is None:
            self._categories = []
        else:
            self._categories = categories
            for cat in categories:
                self._category_dict[cat] = []

        self._docs_per_cat = docs_per_cat
        self._training_ratio = training_ratio
        self._characteristic_num = characteristic_num
        self._silent = silent
        self._metric_type = metric_type

    def _choose_docs(self):
        for cat in self._category_dict.keys():
            # list all docs of the category
            partial_path = os.path.join(self._document_database_path, cat + '/')
            cat_path = os.path.join(os.pardir, partial_path)
            doc_list = os.listdir(cat_path)

            # keep only <min(cat_size,docs_per_cat)> random docs of each category ( or all of them if docs_per_cat==-1 )
            if self._docs_per_cat != -1 and self._docs_per_cat < len(doc_list):
                docs_to_remove = len(doc_list) - self._docs_per_cat

                for _ in range(0, docs_to_remove):
                    del doc_list[randint(0, len(doc_list) - 1)]

            # add them to the doc collection and split them to E and A sets (E will have 0 in the tuple)
            # based on the training ratio
            num_of_e_docs = int(ceil(len(doc_list) * self._training_ratio))
            for i in range(0, len(doc_list)):
                if i < num_of_e_docs:
                    self._category_dict[cat].append((doc_list[i], 0))
                else:
                    self._category_dict[cat].append((doc_list[i], 1))

    def _index_and_extract_characteristics(self):
        # gather all docs of E collection
        docs_to_be_indexed = []
        for cat in self._category_dict.keys():
            for doc in self._category_dict[cat]:
                if doc[1] == 0:
                    docs_to_be_indexed.append(cat + '/' + doc[0])
                else:
                    break

        # perform indexing
        self._total_docs_in_e = len(docs_to_be_indexed)
        indexer = DocumentIndexer(docs_to_be_indexed, self._silent)
        indexer.start()
        # 'characteristics' will have a part of the original index, containing only the top characteristics
        self._characteristics = indexer.extract_top_characteristics(self._characteristic_num)

    def _generate_models(self):
        # we initialize the category model structure. It will be a dictionary with category names as keys
        # and for each one a list of tuples (one tuple for every document already belonging in that category). The tuple
        # will contain in the first place the filename of the document and then the vector for the characteristics
        for cat in self._category_dict.keys():
            self._category_models[cat] = []
            for doc in self._category_dict[cat]:
                if doc[1] == 1:  # we stop at the testing data
                    break
                self._category_models[cat].append((doc[0], [0 for _ in range(self._characteristic_num)]))

        # then we will loop our sliced index, and for every lemma/characteristic..
        self._characteristics_strings = list(self._characteristics.keys())
        count = 0
        for characteristic in self._characteristics_strings:
            index_data_for_characteristic = self._characteristics[characteristic]

            # ..we will look at all the documents that contain it..
            for document in index_data_for_characteristic:
                file_name_parts = document['id'].split('/')
                if len(file_name_parts) == 1:
                    file_name_parts = document['id'].split('\\')

                # and for every single document we will go to its place in our model structure, in order to update the
                # TF-IDF value of this characteristic
                inner_count = 0
                for doc_model in self._category_models[file_name_parts[0]]:
                    if doc_model[0] == file_name_parts[1]:
                        self._category_models[file_name_parts[0]][inner_count][1][count] = document['w']
                        break
                    inner_count += 1

            count += 1

    def train(self):
        if len(self._categories) == 0:
            print('No categories chosen.. Terminating.')
            return

        # first of all we will randomly pick our doc collection (and split it to E, A according to training ratio)
        if not self._silent:
            print('-Choosing E and A sets.. ', end='')
        self._choose_docs()
        if not self._silent:
            print('OK')

        # then we will perform indexing using our DocumentIndexer and extract the characteristics
        if not self._silent:
            print('\n-Performing indexing of E set and extraction of characteristics.. ')
        self._index_and_extract_characteristics()

        # and then we generate the category models using the E documents
        if not self._silent:
            print('\n-Generating category models.. ', end='')
        self._generate_models()
        if not self._silent:
            print('OK')
            print('\n-Training Complete!')

    @staticmethod
    def _remove_closed_class_categories(tagged):
        ret = []
        ban_list = ['CD', 'CC', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT',
                    'WP', 'WP$', 'WRB']
        for term in tagged:
            if term[1] not in ban_list:
                ret.append(term)
        return ret

    def _generate_test_model(self, category, filename, wordnet_lemmatizer):
        test_model = [0 for _ in range(self._characteristic_num)]
        doc_path = os.path.join('../20news-19997/20_newsgroups/', category, filename)

        with open(doc_path, encoding="Latin-1") as f:
            text_raw = "".join([" " if ch in string.punctuation else ch for ch in f.read()])

            text_raw = text_raw.strip()
            content_tokenized = [wordnet_lemmatizer.lemmatize(w.lower()) for w in nltk.tokenize.word_tokenize(text_raw)]
            content_pos_tagged = nltk.pos_tag(content_tokenized)
            content_pos_tagged_no_closed = self._remove_closed_class_categories(content_pos_tagged)

            # calculate tf for the test model
            for term in content_pos_tagged_no_closed:
                for i in range(len(self._characteristics_strings)):
                    if term[0] == self._characteristics_strings[i]:
                        test_model[i] += 1

            # calculate idf from the E data and update test_model list with the tf*idf value
            for i in range(len(self._characteristics_strings)):
                test_model[i] = test_model[i] * log(
                    self._total_docs_in_e / len(self._characteristics[self._characteristics_strings[i]]))

        return test_model

    def test(self):
        wordnet_lemmatizer = WordNetLemmatizer()
        total_tests = 0
        correct_decisions = 0
        for cat in self._category_dict.keys():
            # list all docs of the category
            for doc in self._category_dict[cat]:
                if doc[1] == 0:
                    continue

                total_tests += 1
                if not self._silent:
                    print('-Test #' + str(total_tests) + ' cat: ' + cat + ', doc name: ' + doc[0] + '\n-Generating model.. ', end='')
                model = self._generate_test_model(cat, doc[0], wordnet_lemmatizer)
                if not self._silent:
                    print('OK')

                metrics = VectorMetrics(self._metric_type)

                if not self._silent:
                    print('-Comparing models.. ', end='')
                similarities = {}
                for cat_model in self._category_models.keys():
                    s = 0
                    c = 0
                    for doc_model in self._category_models[cat_model]:
                        c += 1
                        s += metrics.calc(doc_model[1], model)
                    similarities[cat_model] = s / c
                decision = max(similarities, key=similarities.get)

                if decision == cat:
                    correct_decisions += 1
                if not self._silent:
                    print('OK\n-Decision: ' + decision + ' (Accuracy so far: ' + str(correct_decisions*100/total_tests)+'%) \n')

        print('Results: ' + str(correct_decisions)+'/' + str(total_tests) + ' (' + str(correct_decisions*100/total_tests)+'%)')
