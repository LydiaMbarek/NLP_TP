# parsing understanding of the meaning

from __future__ import unicode_literals, print_function
from spacy.training import Example
import plac
import random
import spacy
from pathlib import Path


# training data
# heads: indice of the head of each token
# deps : the syntactic relationship between the token and its head, "dependency label"
TRAIN_DATA = [
    ("All palestinians are martyrdoms", {
        'heads': [1, 2, 2, 2],  # index of token head
        'deps': ['Quantity', 'ATTRIBUTE', 'ROOT', 'ATTRIBUTE']
    }),
    ("The whole world stand with Palestine", {
        'heads': [2, 1, 3, 3, 5, 3],  # index of token head
        'deps': ['-', 'Quantity', 'ATTRIBUTE', 'ROOT', '-', 'LOCATION']
    }),
    ("Few bad people support zionist", {
        'heads': [2, 2, 3, 3, 3],
        'deps': ['Quantity', 'ATTRIBUTE', 'ATTRIBUTE', 'ROOT', 'ATTRIBUTE']
    }),
    ("Half of Gaza is gone", {
        'heads': [2, 2, 3, 3, 3],
        'deps': ['Quantity', '-', 'LOCATION', 'ROOT', '-']
    }),
    ("Many students completed the assignment", {
        'heads': [1, 2, 2, 4, 2],
        'deps': ['Quantity', 'ATTRIBUTE', 'ROOT', '-', 'ATTRIBUTE']
    }),
    ("Some books are on the shelf", {
        'heads': [1, 2, 2, 5, 5, 2],
        'deps': ['Quantity', 'ATTRIBUTE', 'ROOT', '-', '-', 'PLACE']
    }),
    ("Enough crime war was done by zionist", {
        'heads': [1, 3, 1, 3, 3, 6, 3],
        'deps': ['Quantity', 'ATTRIBUTE', 'ATTRIBUTE', 'ROOT', '-', '-', 'ATTRIBUTE']
    }),
    ("Numerous opportunities are available", {
        'heads': [1, 2, 2, 2],
        'deps': ['Quantity', 'ATTRIBUTE', 'ROOT', 'ATTRIBUTE']
    }),
    ("All nations sympathizes with the people of Palestine", {
        'heads': [1, 2, 2, 5, 5, 2, 7, 5],
        'deps': ['Quantity', 'ATTRIBUTE', 'ROOT', '-', '-', 'ATTRIBUTE', '-', 'LOCATION']
    })
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def dependency(model=None, output_dir=None, n_iter=60):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'parser' in nlp.pipe_names:
        nlp.remove_pipe('parser')
    parser = nlp.add_pipe('parser')
    # nlp.add_pipe(parser, first=True)
    for text, annotations in TRAIN_DATA:
        for dep in annotations.get('deps', []):
            parser.add_label(dep)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'parser']
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], sgd=optimizer, losses=losses)
            print(losses)

    # test the trained model
    texts = ["Many nations condemned the attacks on Palestine", "Few people believe in ghosts"]
    docs = nlp.pipe(texts)
    for doc in docs:
        print(doc.text)
        print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != '-'])

    # test_model(nlp)
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        docs = nlp2.pipe(texts)
        for doc in docs:
            print(doc.text)
            print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != '-'])
        # test_model(nlp2)


dependency()
