import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Define english stopwords
stop_words = set(stopwords.words("english") + [".", "!"])

# Define the sentences per the assignment
sent1 = "Very good course!!!"
sent2 = "The teacher is really good."
sent3 = "The teacher and the course are very good."


# Normalize a sentence
def normalize(sent):
    tokens = word_tokenize(sent)

    # Make lowercase
    words = [w.lower() for w in tokens]

    # Remove stopwords
    words = [w for w in words if w not in stop_words]

    # Define the word net lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize the words
    lemmas = [lemmatizer.lemmatize(w) for w in words]

    return lemmas


# Normalize the sentences
sent1 = normalize(sent1)
sent2 = normalize(sent2)
sent3 = normalize(sent3)

# Make ONE text out of the sentences
text = nltk.Text(sent1 + sent2 + sent3)

# Plot the histogram of words
text.plot()

# Make THREE texts out of sentences
text1 = nltk.Text(sent1)
text2 = nltk.Text(sent2)
text3 = nltk.Text(sent3)

# Turn them into a text collection so that nltk can calculate tf and tf*idf for us.
texts = nltk.TextCollection([text1, text2, text3])

# Calculate tf and tf*idf
vec = [
    {
        "term": term,
        "text": _text.name,
        "tf": texts.tf(term, _text),
        "tf_idf": texts.tf_idf(term, _text),
    }
    for _text in [text1, text2, text3]
    for term in _text
]
print(vec)
