import numpy as np

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

c = CountVectorizer()

X = c.fit_transform(open('data/corpus_sentence.txt', 'r', encoding='utf8'))

with open('data/corpus_tags.txt', 'r', encoding='utf8') as f:
    y = [_.strip() for _ in f.readlines()]

C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)

test_sentence = ['yo see you!', 'see ya!', 'good to see you']

result_set = svc.predict(c.transform(test_sentence))

for test, result in zip(test_sentence, result_set):
    print(test, ' : ', result)

print(accuracy_score(['when-bye', 'when-bye', 'when-meet'], result_set))
