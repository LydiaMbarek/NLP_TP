import plac
import random
from pathlib import Path
import spacy
from spacy.training import Example

# ******************** TRAINING POS TAGGER ************************ #
print("********************* TRAINING POS TAGGER ****************************")

TAG_MAP = {
    "N": {"pos": "NOUN"},
    "V": {"pos": "VERB"},
    "J": {"pos": "ADJ"},
    "PRON": {"pos": "PRON"},
    "ADV": {"pos": "ADV"},
    "DET": {"pos": "DET"},
    "CONJ": {"pos": "CONJ"},
    "NNS": {"pos": "NOUN", "morph": {"number": "plur"}},
}

TRAIN_DATA = [
    ("I like green apples", {"tags": ["N", "V", "J", "NNS"]}),
    ("Eat blue apples", {"tags": ["V", "J", "NNS"]}),
    ("I like green apples", {"tags": ["N", "V", "J", "NNS"]}),
    ("Eat blue apples", {"tags": ["V", "J", "NNS"]}),
    ("She sells seashells", {"tags": ["N", "V", "NNS"]}),
    ("The birds sing", {"tags": ["DET", "NNS", "V"]}),
    ("They are dancing", {"tags": ["PRON", "V", "V"]}),
    ("We eat cake", {"tags": ["PRON", "V", "N"]}),
    ("The black cats", {"tags": ["DET", "J", "NNS"]}),
    ("The quick brown fox jumps", {"tags": ["DET", "J", "J", "N", "V"]}),
    ("Running fast", {"tags": ["V", "ADV"]}),
    ("Oranges and strawberries are fruits", {"tags": ["NNS", "CONJ", "NNS", "V", "NNS"]}),
    ("They walk slowly", {"tags": ["PRON", "V", "ADV"]}),
    ("The cat and the dog play", {"tags": ["DET", "N", "CONJ", "DET", "N", "V"]}),
    ("Beautiful flowers bloom", {"tags": ["J", "NNS", "V"]}),
]


@plac.annotations(
    lang=("ISO Code of language to use", "option", "l", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def pos_tag(lang="en", output_dir=None, n_iter=50):
    nlp = spacy.blank(lang)
    if "tagger" not in nlp.pipe_names:
        tagger = nlp.add_pipe("tagger")
        for label, values in TAG_MAP.items():
            tagger.add_label(label)

    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        print(losses)

    test_text = "cats and dogs are animals"
    doc = nlp(test_text)
    print("Tags", [(t.text, t.tag_, t.pos_) for t in doc])

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        # Test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc = nlp2(test_text)
        print("Tags", [(t.text, t.tag_, t.pos_) for t in doc])


pos_tag()

# ******************** TRAINING NER TAGGER ************************ #
print("********************* TRAINING NER TAGGER ****************************")

LABEL = "COLOR"

TRAIN_DATA2 = [
    ("the sky is blue", {"entities": [(11, 15, LABEL)]}),
    ("grass is green", {"entities": [(9, 14, LABEL)]}),
    ("the sun is yellow", {"entities": [(11, 17, LABEL)]}),
    ("roses are red", {"entities": [(10, 13, LABEL)]}),
    ("the ocean is blue", {"entities": [(13, 17, LABEL)]}),
    ("violets are violet", {"entities": [(12, 18, LABEL)]}),
    ("the sunflowers are yellow", {"entities": [(19, 25, LABEL)]}),
    ("the apple is red", {"entities": [(13, 16, LABEL)]}),
    ("the banana is yellow", {"entities": [(14, 20, LABEL)]}),
    ("the grass is green", {"entities": [(13, 18, LABEL)]}),
    ("My eyes are green", {"entities": [(12, 17, LABEL)]}),
    ("I have a black hair", {"entities": [(9, 14, LABEL)]}),
    ("My sister has a white hair", {"entities": [(16, 21, LABEL)]}),
    ("The flag of Mouloudia club has two colors, red and green", {"entities": [(43, 46, LABEL)] and [(51, 56, LABEL)]}),
    ("I had an orange dress", {"entities": [(9, 15, LABEL)]}),
]


@plac.annotations(
    lang=("ISO Code of language to use", "option", "l", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def ner_tagger(lang="en", output_dir=None, n_iter=50):
    nlp = spacy.blank(lang)
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    ner.add_label(LABEL)

    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(TRAIN_DATA2)
        losses = {}
        for text, annotations in TRAIN_DATA2:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], sgd=optimizer, drop=0.5, losses=losses)
        print(losses)

    test_text = "my flowers are red and the sky is blue and I eat an orange apple"
    doc = nlp(test_text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        # Test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc = nlp2(test_text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])


ner_tagger()
