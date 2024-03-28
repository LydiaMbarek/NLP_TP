import spacy
from nltk.corpus import reuters, gutenberg
from gensim import corpora
from gensim import models
nlp = spacy.load("en_core_web_sm")


documents = reuters.sents(categories=['money-supply'])[:200]
# documents = gutenberg.sents('carroll-alice.txt')[:100]

# cleaning data
texts = []
for doc in documents:
    text = []
    for sentence in doc:
        doc = nlp(sentence)
        for w in doc:
            if not w.is_stop and not w.is_punct and not w.like_num:
                text.append(w.lemma_)
    texts.append(text)

print(texts)

# *********************** BI_GRAM ****************************** #
print('**************** BI_GRAM ****************************')

# Bi_gram processing
bigram = models.Phrases(texts, min_count=3, threshold=3)
texts2 = [bigram[line] for line in texts]
print(texts2)

# faire le dictionnaire pour le bi_gram
dictionary = corpora.Dictionary(texts2)
print(dictionary.token2id)

# corpus pour bi_gram (combien de fois pour chaque id de mot dans un document)
corpus = [dictionary.doc2bow(text) for text in texts2]
print(corpus)

# TF-IDF of a Bi_gram
tfidf = models.TfidfModel(corpus)

for document in tfidf[corpus]:
    print(document)

# print extracted Bi_gram
print("\nExtracted bigrams:")
for phrase in bigram.export_phrases():
    print(phrase)

# *********************** TRI_GRAM ****************************** #
print('**************** TRIGRAM ****************************')

# Tri_gram processing for the bi_gram, this allows to capture higher level associations between words
trigram = models.Phrases(texts2, min_count=3, threshold=3)
texts3 = [trigram[line] for line in texts2]
print(texts3)

# faire le dictionnaire pour le tri_gram
dictionary2 = corpora.Dictionary(texts3)
print(dictionary2.token2id)

# corpus pour tri_gram (combien de fois pour chaque id de mot dans un document)
corpus2 = [dictionary2.doc2bow(text) for text in texts3]
print(corpus2)

# TF-IDF of a Tri_gram
tfidf2 = models.TfidfModel(corpus2)

for document in tfidf2[corpus2]:
    print(document)

# print extracted trigram
print("\nExtracted Trigrams:")
for phrase in trigram.export_phrases():
    print(phrase)
