import os
from random import randint
from classifier.docclassifier import DocClassifier


def choose_random_categories(num):
    used = []
    ret = []
    while len(ret) < num:
        sel = randint(1, 20) - 1
        if sel not in used:
            used.append(sel)
            ret.append(sel)
    return ret


if __name__ == '__main__':
    print('=== Document Classification ===')
    untagged = [f for f in os.listdir(os.path.abspath(os.path.join(os.pardir, '20news-19997/20_newsgroups')))]
    print('-Enter the number of categories that will be randomly chosen')
    cat_num = int(input())
    print('\n-The following categories were chosen:')
    cat_ids = choose_random_categories(cat_num)

    chosen_cats = []
    for i in cat_ids:
        print(untagged[i])
        chosen_cats.append(untagged[i])
    print('')
    print('-Enter the number of documents to randomly choose from each category (total in E (training data) + A (testing data) sets), or -1 to input all docs')
    num_docs = int(input())

    print('-Enter the training ratio in decimal form (e.g  0.6 = 60%->E , 40%->A )')
    ratio = float(input())

    print('-Enter the number of characteristics')
    chars = int(input())

    print('-Choose vector metric. Type:\n1) for Cosine similarity\n2) for Jaccard Index')
    metric = int(input())
    if metric != 1 and metric != 2:
        metric = 1

    # actual entry point of algorithm
    dc = DocClassifier(categories=chosen_cats, docs_per_cat=num_docs, training_ratio=ratio, characteristic_num=chars, metric_type=metric)
    dc.train()
    dc.test()
