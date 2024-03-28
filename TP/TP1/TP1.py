import spacy
from collections import Counter
from nltk.stem.porter import PorterStemmer
# from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

# ************************** HOMEWORK ************************* #
complete_text = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech company. He is"
    " interested in learning Natural Language Processing."
    " There is a developer conference happening on 21 July"
    ' 2019 in London. It is titled "Applications of Natural'
    ' Language Processing". There is a helpline number'
    " available at +44-1234567891. Gus is helping organize it."
    " He keeps organizing local Python meetups and several"
    " internal talks at his workplace. Gus is also presenting"
    ' a talk. The talk will introduce the reader about "Use'
    ' cases of Natural Language Processing in Fintech".'
    " Apart from his work, he is very passionate about music."
    " Gus is learning to play the Piano. He has enrolled"
    " himself in the weekend batch of Great Piano Academy."
    " Great Piano Academy is situated in Mayfair or the City"
    " of London and has world-class piano instructors."
)


# complete_doc = nlp(complete_text)
# print([(token.text, token.idx) for token in complete_doc])


def preprocessing(text):
    # lower casing
    text = text.lower()

    # doc object
    text = nlp(text)

    # Removal of punctuations and stopwords
    text_up = [token for token in text if not token.is_stop and not token.is_punct]

    # Removal of frequent words and rare words
    word_counter = Counter(text_up)  # calculate the frequency of each word
    freq_words = [word for word, freq in word_counter.most_common(10)]
    rare_words = [word for word, freq in
                  word_counter.most_common()[:-10 - 1:-1]]  # the 10 element from the end, in reverse order
    text_up = [token for token in text_up if token not in freq_words and token not in rare_words]
    print(text_up)

    # lemmatisation ans stemming
    lemma_words = []
    for token in text_up:
        token = [token.lemma_][0]  # lemmatisation
        token = PorterStemmer().stem(token)  # spacy doesn't have a function of stemming
        lemma_words.append(token)

    return lemma_words


mol = preprocessing(complete_text)
print(mol)
