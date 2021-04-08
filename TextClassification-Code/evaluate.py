import math

from build import models, reader
from build import labels as categories
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

docs = reader.fileids(categories=categories)
labels = [reader.categories(fileids=[fid])[0] for fid in docs]

train_docs, test_docs, train_labels, test_labels = tts(docs, labels, test_size=0.3)


def get_docs(fids):
    for fid in fids:
        yield list(reader.docs(fileids=[fid]))


for model in models:
    name = model.named_steps['classifier'].__class__.__name__
    if 'reduction' in model.named_steps:
        name += " (Truncated)"
    model.fit(get_docs(train_docs), train_labels)
    pred = model.predict(get_docs(test_docs))
    print(name,'\n',classification_report(test_labels, pred, labels=categories))
    plot_confusion_matrix(model,get_docs(test_docs),test_labels)
    plt.title(name)
    plt.show()
