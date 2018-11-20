"""This script is experimental.

Try pre-training an LSTM with the 'LMAO' objective. Requires PyTorch.
"""
from __future__ import print_function, unicode_literals
import plac
import random
import numpy
import time
import ujson as json
from pathlib import Path
import sys
from collections import Counter

from spacy.tokens.doc import Doc
import spacy
from spacy.attrs import ID, HEAD
from spacy.util import minibatch, minibatch_by_words, use_gpu, compounding, ensure_path
from spacy._ml import Tok2Vec, flatten, chain, zero_init, create_default_optimizer
from thinc.v2v import Affine
from thinc.t2t import ExtractWindow

from thinc.v2v import Model, Maxout, Softmax, Affine, ReLu
from thinc.i2v import HashEmbed
from thinc.misc import LayerNorm as LN
from thinc.api import add, layerize, chain, clone, concatenate, with_flatten
from thinc.api import FeatureExtracter, with_getitem, flatten_add_lengths
from thinc.api import with_square_sequences, wrap, noop

from spacy.attrs import ID, ORTH, LOWER, NORM, PREFIX, SUFFIX, SHAPE
from spacy._ml import Tok2Vec_LSTM

try:
    import torch.nn
    from thinc.extra.wrappers import PyTorchWrapperRNN
except:
    torch = None


def prefer_gpu():
    used = spacy.util.use_gpu(0)
    if used is None:
        return False
    else:
        import cupy.random

        cupy.random.seed(0)
        if torch is not None:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        return True


def load_texts(path):
    """Load inputs from a jsonl file.
    
    Each line should be a dict like {"text": "..."}
    """
    path = ensure_path(path)
    with path.open("r", encoding="utf8") as file_:
        texts = [json.loads(line) for line in file_]
    random.shuffle(texts)
    return texts


def stream_texts():
    for line in sys.stdin:
        yield json.loads(line)


def make_update(model, docs, optimizer, drop=0.0):
    """Perform an update over a single batch of documents.

    docs (iterable): A batch of `Doc` objects.
    drop (float): The droput rate.
    optimizer (callable): An optimizer.
    RETURNS loss: A float for the loss.
    """
    docs = [doc for doc in docs if len(doc) >= 1]
    if not docs:
        return 0.0
    predictions, backprop = model.begin_update(docs, drop=drop)
    loss, gradients = get_lm_vectors_loss(
        model.ops, docs, predictions)
    backprop(gradients, sgd=optimizer)
    return loss


def get_lm_vectors_loss(ops, docs, prediction):
    """Compute a mean-squared error loss between the documents' vectors and
    the prediction.    

    Note that this is ripe for customization! We could compute the vectors
    in some other word, e.g. with an LSTM language model, or use some other
    type of objective.
    """
    # The simplest way to implement this would be to vstack the
    # token.vector values, but that's a bit inefficient, especially on GPU.
    # Instead we fetch the index into the vectors table for each of our tokens,
    # and look them up all at once. This prevents data copying.
    padding = numpy.asarray([0], dtype="uint64")
    ids = ops.flatten(
        [padding] + [doc.to_array(ID).ravel() for doc in docs] + [padding]
    )
    vectors = docs[0].vocab.vectors.data[ids]
    target = ops.xp.hstack((vectors[2:], vectors[:-2]))
    d_scores = (prediction - target) / prediction.shape[0]
    # Don't want to return a cupy object here
    loss = float((d_scores ** 2).sum())
    return loss, d_scores


def create_pretraining_model(nlp, tok2vec, objective="basic"):
    """Define a network for the pretraining."""
    output_size = nlp.vocab.vectors.data.shape[1]
    # This is annoying, but the parser etc have the flatten step after
    # the tok2vec. To load the weights in cleanly, we need to match
    # the shape of the models' components exactly. So what we cann
    # "tok2vec" has to be the same set of processes as what the components do.
    with Model.define_operators({">>": chain, "|": concatenate}):

        l2r_model = (
            tok2vec.l2r
            >> flatten
            >> LN(Maxout(output_size, tok2vec.l2r.nO, pieces=3))
            >> zero_init(Affine(output_size, drop_factor=0.0))
        )
        r2l_model = (
            tok2vec.r2l
            >> flatten
            >> LN(Maxout(output_size, tok2vec.r2l.nO, pieces=3))
            >> zero_init(Affine(output_size, drop_factor=0.0))
        )

        model = tok2vec.embed >> (l2r_model | r2l_model)

    model.tok2vec = tok2vec
    model.begin_training([nlp.make_doc("Give it a doc to infer shapes")])
    tok2vec.begin_training([nlp.make_doc("Give it a doc to infer shapes")])
    tokvecs = tok2vec([nlp.make_doc('hello there'), nlp.make_doc(u'and hello')])
    print(tokvecs.shape)
    return model


def make_docs(nlp, batch, heads=True):
    docs = []
    for record in batch:
        text = record["text"]
        if "tokens" in record:
            doc = Doc(nlp.vocab, words=record["tokens"])
        else:
            doc = nlp.make_doc(text)
        if "heads" in record:
            heads = record["heads"]
            heads = numpy.asarray(heads, dtype="uint64")
            heads = heads.reshape((len(doc), 1))
            doc = doc.from_array([HEAD], heads)
        if len(doc) >= 1 and len(doc) < 200:
            docs.append(doc)
    return docs


class ProgressTracker(object):
    def __init__(self, frequency=100000):
        self.loss = 0.0
        self.prev_loss = 0.0
        self.nr_word = 0
        self.words_per_epoch = Counter()
        self.frequency = frequency
        self.last_time = time.time()
        self.last_update = 0

    def update(self, epoch, loss, docs):
        self.loss += loss
        words_in_batch = sum(len(doc) for doc in docs)
        self.words_per_epoch[epoch] += words_in_batch
        self.nr_word += words_in_batch
        words_since_update = self.nr_word - self.last_update
        if words_since_update >= self.frequency:
            wps = words_since_update / (time.time() - self.last_time)
            self.last_update = self.nr_word
            self.last_time = time.time()
            loss_per_word = self.loss - self.prev_loss
            status = (
                epoch,
                self.nr_word,
                "%.5f" % self.loss,
                "%.4f" % loss_per_word,
                int(wps),
            )
            self.prev_loss = float(self.loss)
            return status
        else:
            return None


@plac.annotations(
    texts_loc=("Path to jsonl file with texts to learn from", "positional", None, str),
    vectors_model=("Name or path to vectors model to learn from"),
    output_dir=("Directory to write models each epoch", "positional", None, str),
    width=("Width of CNN layers", "option", "cw", int),
    depth=("Depth of CNN layers", "option", "cd", int),
    embed_rows=("Embedding rows", "option", "er", int),
    dropout=("Dropout", "option", "d", float),
    seed=("Seed for random number generators", "option", "s", float),
    nr_iter=("Number of iterations to pretrain", "option", "i", int),
)
def pretrain(
    texts_loc,
    vectors_model,
    output_dir,
    width=128,
    depth=4,
    embed_rows=1000,
    dropout=0.2,
    nr_iter=10,
    seed=0,
):
    """
    Pre-train the 'token-to-vector' (tok2vec) layer of pipeline components,
    using an approximate language-modelling objective. Specifically, we load
    pre-trained vectors, and train a component like a CNN, BiLSTM, etc to predict
    vectors which match the pre-trained ones. The weights are saved to a directory
    after each epoch. You can then pass a path to one of these pre-trained weights
    files to the 'spacy train' command.

    This technique may be especially helpful if you have little labelled data.
    However, it's still quite experimental, so your mileage may vary.

    To load the weights back in during 'spacy train', you need to ensure
    all settings are the same between pretraining and training. The API and
    errors around this need some improvement.
    """
    config = dict(locals())
    output_dir = ensure_path(output_dir)
    random.seed(seed)
    numpy.random.seed(seed)
    if not output_dir.exists():
        output_dir.mkdir()
    with (output_dir / "config.json").open("w") as file_:
        file_.write(json.dumps(config))
    has_gpu = prefer_gpu()
    nlp = spacy.load(vectors_model)
    tok2vec = Tok2Vec_LSTM(width, embed_rows, depth, dropout)
    print(dir(tok2vec))
    model = create_pretraining_model(nlp, tok2vec)
    optimizer = create_default_optimizer(model.ops)
    tracker = ProgressTracker()
    print("Epoch", "#Words", "Loss", "L/W", "w/s")
    texts = stream_texts() if texts_loc == "-" else load_texts(texts_loc)
    for epoch in range(nr_iter):
        for batch in minibatch(texts, size=256):
            docs = make_docs(nlp, batch, heads=False)
            loss = make_update(model, docs, optimizer, drop=dropout)
            progress = tracker.update(epoch, loss, docs)
            if progress:
                print(*progress)
                if texts_loc == "-" and tracker.words_per_epoch[epoch] >= 10 ** 6:
                    break
        with (output_dir / ("model%d.bin" % epoch)).open("wb") as file_:
            # This is annoying -- work around how Parser expects this
            file_.write(chain(tok2vec, layerize(noop())).to_bytes())
        with (output_dir / "log.jsonl").open("a") as file_:
            file_.write(
                json.dumps(
                    {"nr_word": tracker.nr_word, "loss": tracker.loss, "epoch": epoch}
                )
            )
        if texts_loc != "-":
            texts = load_texts(texts_loc)


if __name__ == "__main__":
    plac.call(pretrain)
