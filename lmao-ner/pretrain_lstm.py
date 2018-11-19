'''This script is experimental.

Try pre-training an LSTM with the 'LMAO' objective. Requires PyTorch.
'''
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
from thinc.api import with_square_sequences

from spacy.attrs import ID, ORTH, LOWER, NORM, PREFIX, SUFFIX, SHAPE

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
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return True


def load_texts(path):
    '''Load inputs from a jsonl file.
    
    Each line should be a dict like {"text": "..."}
    '''
    path = ensure_path(path)
    with path.open('r', encoding='utf8') as file_:
        texts = [json.loads(line) for line in file_]
    random.shuffle(texts)
    return texts

def stream_texts():
    for line in sys.stdin:
        yield json.loads(line)


def make_update(model, docs, optimizer, drop=0.):
    """Perform an update over a single batch of documents.

    docs (iterable): A batch of `Doc` objects.
    drop (float): The droput rate.
    optimizer (callable): An optimizer.
    RETURNS loss: A float for the loss.
    """
    docs = [doc for doc in docs if len(doc) >= 1]
    if not docs:
        return 0.
    predictions, backprop = model.begin_update(docs, drop=drop)
    loss, gradients = get_lm_vectors_loss(model.ops, docs, predictions, direction='forward')
    backprop(gradients, sgd=optimizer)
    return loss


def get_lm_vectors_loss(ops, docs, prediction, direction='forward'):
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
    padding = numpy.asarray([0], dtype='uint64')
    ids = ops.flatten([doc.to_array(ID).ravel() for doc in docs] + [padding])
    ids = ids[1:]
    assert len(ids) == prediction.shape[0]
    target = docs[0].vocab.vectors.data[ids]
    d_scores = (prediction - target) / prediction.shape[0]
    # Don't want to return a cupy object here
    loss = float((d_scores**2).sum())
    return loss, d_scores


def create_pretraining_model(nlp, tok2vec, objective='basic'):
    '''Define a network for the pretraining. We simply add an output layer onto
    the tok2vec input model. The tok2vec input model needs to be a model that
    takes a batch of Doc objects (as a list), and returns a list of arrays.
    Each array in the output needs to have one row per token in the doc.
    '''
    if objective == 'head_vectors':
        output_size = nlp.vocab.vectors.data.shape[1] * 2
    elif objective == 'trigrams':
        output_size = nlp.vocab.vectors.data.shape[1] * 3
    else:
        output_size = nlp.vocab.vectors.data.shape[1]
    output_layer = chain(
        LN(Maxout(output_size, tok2vec.nO, pieces=3)),
        zero_init(Affine(output_size, drop_factor=0.0))
    )
    # This is annoying, but the parser etc have the flatten step after
    # the tok2vec. To load the weights in cleanly, we need to match
    # the shape of the models' components exactly. So what we cann
    # "tok2vec" has to be the same set of processes as what the components do.
    tok2vec = chain(tok2vec, flatten)
    model = chain(
        tok2vec,
        output_layer
    )
    model.tok2vec = tok2vec
    model.output_layer = output_layer
    model.begin_training([nlp.make_doc('Give it a doc to infer shapes')])
    return model


def PyTorchLSTM(nO, nI, depth, dropout=0.2):
    if depth == 0:
        return layerize(noop())
    model = torch.nn.LSTM(nI, nO, depth, dropout=dropout)
    return with_square_sequences(PyTorchWrapperRNN(model))


def Tok2Vec_LSTM(width, embed_size, depth, dropout):
    cols = [ID, NORM, PREFIX, SUFFIX, SHAPE, ORTH]
    with Model.define_operators({'>>': chain, '|': concatenate, '**': clone,
                                 '+': add}):
        norm = HashEmbed(width, embed_size, column=cols.index(NORM),
                         name='embed_norm')
        prefix = HashEmbed(width, embed_size//2, column=cols.index(PREFIX),
                            name='embed_prefix')
        suffix = HashEmbed(width, embed_size//2, column=cols.index(SUFFIX),
                            name='embed_suffix')
        shape = HashEmbed(width, embed_size//2, column=cols.index(SHAPE),
                            name='embed_shape')
        embed = (
            (norm | prefix | suffix | shape)
            >> LN(Maxout(width, width*4, pieces=3))
        )
        model = (
            FeatureExtracter([ORTH, LOWER, PREFIX, SUFFIX, SHAPE, ID])
            >> with_flatten(embed, pad=1)
            >> PyTorchLSTM(width, width, depth, dropout=dropout)
        )
        # Work around thinc API limitations :(. TODO: Revise in Thinc 7
        model.nO = width
        model.embed = embed
    return model


def make_docs(nlp, batch, heads=True):
    docs = []
    for record in batch:
        text = record['text']
        if 'tokens' in record:
            doc = Doc(nlp.vocab, words=record['tokens'])
        else:
            doc = nlp.make_doc(text)
        if 'heads' in record:
            heads = record['heads']
            heads = numpy.asarray(heads, dtype='uint64')
            heads = heads.reshape((len(doc), 1))
            doc = doc.from_array([HEAD], heads)
        if len(doc) >= 1 and len(doc) < 200:
            docs.append(doc)
    return docs


class ProgressTracker(object):
    def __init__(self, frequency=100000):
        self.loss = 0.
        self.prev_loss = 0.
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
            loss_per_word = (self.loss - self.prev_loss) 
            status = (epoch, self.nr_word, '%.5f' % self.loss,
                     '%.4f' % loss_per_word, int(wps))
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
def pretrain(texts_loc, vectors_model, output_dir, width=128, depth=4,
        embed_rows=1000, dropout=0.2, nr_iter=10, seed=0):
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
    with (output_dir / 'config.json').open('w') as file_:
        file_.write(json.dumps(config))
    has_gpu = prefer_gpu()
    nlp = spacy.load(vectors_model)
    model = create_pretraining_model(nlp, Tok2Vec_LSTM(width, embed_rows, depth, dropout))
    optimizer = create_default_optimizer(model.ops)
    tracker = ProgressTracker()
    print('Epoch', '#Words', 'Loss', 'L/W', 'w/s')
    texts = stream_texts() if texts_loc == '-' else load_texts(texts_loc) 
    for epoch in range(nr_iter):
        for batch in minibatch(texts, size=256):
            #docs = [nlp.make_doc(text) for text in batch]
            docs = make_docs(nlp, batch, heads=False)
            loss = make_update(model, docs, optimizer, drop=dropout)
            progress = tracker.update(epoch, loss, docs)
            if progress:
                print(*progress)
                if texts_loc == '-' and tracker.words_per_epoch[epoch] >= 10**6:
                    break
        with (output_dir / ('model%d.bin' % epoch)).open('wb') as file_:
            file_.write(model.tok2vec.to_bytes())
        with (output_dir / 'log.jsonl').open('a') as file_:
            file_.write(json.dumps({'nr_word': tracker.nr_word,
                'loss': tracker.loss, 'epoch': epoch}))
        if texts_loc != '-':
            texts = load_texts(texts_loc)

if __name__ == '__main__':
    plac.call(pretrain)
