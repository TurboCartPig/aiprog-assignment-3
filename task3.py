from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from build import create_pipeline
from build import labels as categories
from build import loader, reader

# Import and slit the corpus
docs = reader.fileids(categories=categories)
labels = [reader.categories(fileids=[fid])[0] for fid in docs]

train_docs, test_docs, train_labels, test_labels = train_test_split(
    docs, labels, test_size=0.3
)


def get_docs(fids):
    for fid in fids:
        yield list(reader.docs(fileids=[fid]))


# Define pipelines that apply normalization before the specified models are run
models = [
    ("SVM", create_pipeline(SVC(), False)),
    ("Decision Tree", create_pipeline(DecisionTreeClassifier(), False)),
    ("Random Forest", create_pipeline(RandomForestClassifier(), False)),
]


# Train the models and score them per class
for name, model in models:
    model.fit(get_docs(train_docs), train_labels)
    preds = model.predict(get_docs(test_docs))
    report = classification_report(test_labels, preds, labels=categories)
    print(name)
    print(report)


# Perfome 5-fold cross-validation based TextClassification
for name, model in models:
    scores = {
        "f1": [],
        "cm": [],
    }

    for train_docs, test_docs, train_labels, test_labels in loader:
        model.fit(train_docs, train_labels)
        preds = model.predict(test_docs)

        scores["f1"].append(f1_score(test_labels, preds, average="weighted"))
        scores["cm"].append(confusion_matrix(test_labels, preds))

    print(name)
    print(scores["f1"])
    print(scores["cm"])
