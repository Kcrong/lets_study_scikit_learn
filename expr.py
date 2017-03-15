from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from data_manager import load_dataset

c = CountVectorizer()

sentences, labels = load_dataset('private_data/corpus.csv')

total_len = len(sentences)
train_len = int(total_len * 0.8)

# make feature label
c.fit_transform(sentences)

x_train_raw = sentences[:train_len]
x_train = c.transform(x_train_raw)

x_test_raw = sentences[train_len:]
x_test = c.transform(x_test_raw)

y_train = labels[:train_len]
y_test = labels[train_len:]


C = 100  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)


result_set = svc.predict(x_test)

for test, result in zip(x_test_raw, result_set):
    print(test, ' : ', result)

print(accuracy_score(y_test, result_set))
