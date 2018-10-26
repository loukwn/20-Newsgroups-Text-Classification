import json
import operator
import os
from math import log

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import string


class DocumentIndexer:
    _inv_idx_path = '../data'
    _tagged_docs_path = '../data/tagged_docs'
    _document_database_path = '../20news-19997/20_newsgroups/'
    _index = {}

    def __init__(self, document_list_paths=None, silent=False):
        if document_list_paths is None:
            self._document_list_paths = []
        else:
            self._document_list_paths = document_list_paths
        self._silent = silent

        if not os.path.exists(self._inv_idx_path):
            os.makedirs(self._inv_idx_path)

        if not os.path.exists(self._tagged_docs_path):
            os.makedirs(self._tagged_docs_path)

    def _get_text_from_original(self, filename):
        doc_path = os.path.join(self._document_database_path, filename)
        with open(doc_path, encoding="Latin-1") as f:
            return "".join([" " if ch in string.punctuation else ch for ch in f.read()])

    @staticmethod
    def _remove_closed_class_categories(tagged):
        ret = []
        ban_list = ['CD', 'CC', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT',
                    'WP', 'WP$', 'WRB']
        for term in tagged:
            if term[1] not in ban_list:
                ret.append(term)
        return ret

    @staticmethod
    def _add_content_to_index(index, content, filename):
        for term in content:
            if term[0] not in index:
                index[term[0]] = [{'id': filename, 'total': 1}]
            else:
                found = False
                for i in index[term[0]]:
                    if filename == i['id']:
                        i['total'] = i['total'] + 1
                        found = True
                        break

                if not found:
                    index[term[0]].append({'id': filename, 'total': 1})
        return index

    def _save_inv_index_file(self, index):
        doc_path = os.path.join(self._inv_idx_path, 'inv_index.json')
        with open(doc_path, 'w') as f:
            return json.dump(index, f)

    @staticmethod
    def _update_index(index, n):
        for key in index:
            total = len(index[key])
            for i in range(0, total):
                index[key][i]['w'] = index[key][i]['total'] * log(n / total)
                index[key][i].pop('total', None)
        return index

    def _save_post_tagged_file(self, content, filename):
        file_name_parts = filename.split('/')
        if len(file_name_parts) == 1:
            file_name_parts = filename.split('\\')

        save_path = os.path.join(self._tagged_docs_path, file_name_parts[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        final_path = os.path.join(save_path, file_name_parts[1])
        with open(final_path, 'w') as f:
            for term in content:
                f.write(term[0]+' ')

    def extract_top_characteristics(self, num):
        if not self._silent:
            print('Creating characteristics.. ', end='')
        characteristics = {}

        for lemma in self._index:
            characteristics[lemma] = 0
            for i in self._index[lemma]:
                characteristics[lemma] += i['w']
        sorted_x = sorted(characteristics.items(), key=operator.itemgetter(1), reverse=True)

        to_return = {}
        for i in range(num):
            to_return[sorted_x[i][0]] = self._index[sorted_x[i][0]]

        if not self._silent:
            print('OK')

        return to_return

    def start(self):
        wordnet_lemmatizer = WordNetLemmatizer()
        total_docs = len(self._document_list_paths)
        if total_docs == 0:
            print('No docs chosen to index..')
            return

        count = 0
        for file in self._document_list_paths:
            s = str(count)

            content = self._get_text_from_original(file).strip()
            content_tokenized = [wordnet_lemmatizer.lemmatize(w.lower()) for w in nltk.tokenize.word_tokenize(content)]
            content_pos_tagged = nltk.pos_tag(content_tokenized)
            content_pos_tagged_no_closed = self._remove_closed_class_categories(content_pos_tagged)
            # self._save_post_tagged_file(content_pos_tagged_no_closed, file)
            self._index = self._add_content_to_index(self._index, content_pos_tagged_no_closed, file)

            if not self._silent:
                print('\r[' + s + '/' + str(total_docs) + '] Tagged and added to index: ' + file, end='')
            count += 1

        if not self._silent:
            print('\r[' + str(total_docs) + '/' + str(total_docs) + '] All tagged! ', end='')
            print('\nUpdating index.. ', end='')
        self._index = self._update_index(self._index, total_docs)
        # self._save_inv_index_file(self._index)
        if not self._silent:
            print('OK')
