'''This script is experimental.

Try pre-training the CNN component of the text categorizer using a cheap
language modelling-like objective. Specifically, we load pre-trained vectors
(from something like word2vec, GloVe, FastText etc), and use the CNN to
predict the tokens' pre-trained vectors. This isn't as easy as it sounds:
we're not merely doing compression here, because heavy dropout is applied,
including over the input words. This means the model must often (50% of the time)
use the context in order to predict the word.

To evaluate the technique, we're pre-training with the 50k texts from the IMDB
corpus, and then training with only 100 labels. Note that it's a bit dirty to
pre-train with the development data, but also not *so* terrible: we're not using
the development labels, after all --- only the unlabelled text.
'''
import plac
import random
import spacy
import thinc.extra.datasets
from spacy.util import minibatch
from spacy._ml import Tok2Vec
from spacy.pipeline import TextCategorizer
import numpy
import polyaxon_helper
from polyaxon_helper import send_metrics


class EarlyStopping(object):
    def __init__(self, metric, patience):
        self.metric = metric
        self.max_patience = patience
        self.current_patience = patience
        # We set a minimum best, so that we lose patience with terrible configs.
        self.best = 0.5

    def update(self, result):
        if result[self.metric] >= self.best:
            self.best = result[self.metric]
            self.current_patience = self.max_patience
            return False
        else:
            self.current_patience -= 1
            return self.current_patience <= 0


def report_progress(epoch, best, losses, scores):
    print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
        .format(losses['textcat'], scores['textcat_p'],
        scores['textcat_r'], scores['textcat_f']))
    send_metrics(
        epoch=epoch,
        best_acc=best,
        loss=losses['textcat'],
        P=scores['textcat_p'],
        R=scores['textcat_r'],
        F=scores['textcat_f'])


def load_texts(limit=0):
    train, dev = thinc.extra.datasets.imdb()
    train_texts, train_labels = zip(*train)
    dev_texts, dev_labels = zip(*train)
    train_texts = list(train_texts)
    dev_texts = list(dev_texts)
    random.shuffle(train_texts)
    random.shuffle(dev_texts)
    if limit >= 1:
        return train_texts[:limit]
    else:
        return list(train_texts) + list(dev_texts)


def load_textcat_data(is_final_result=False, limit=0):
    """Load data from the IMDB dataset."""
    train_data, eval_data = thinc.extra.datasets.imdb()
    random.shuffle(train_data)
    if not is_final_result:
        # Partition off part of the train data for evaluation
        eval_data = train_data[-5000:]
        train_data = train_data[:-5000]
    train_texts, train_labels = zip(*train_data)
    eval_texts, eval_labels = zip(*eval_data)
    cats = [{'POSITIVE': bool(y), 'NEGATIVE': not bool(y)} for y in labels]
    eval_cats = [{'POSITIVE': bool(y), 'NEGATIVE': not bool(y)} for y in eval_labels]
    train_data = train_data[-limit:]
    return (texts, cats), (eval_texts, eval_cats)


def prefer_gpu():
    used = spacy.util.use_gpu(0)
    if used is None:
        return False
    else:
        import cupy.random
        cupy.random.seed(0)
        return True


def build_textcat_model(tok2vec, nr_class, width):
    from thinc.v2v import Model, Softmax
    from thinc.api import flatten_add_lengths, chain
    from thinc.t2v import Pooling, mean_pool

    with Model.define_operators({'>>': chain}):
        model = (
            tok2vec
            >> flatten_add_lengths
            >> Pooling(mean_pool)
            >> Softmax(nr_class, width)
        )
    model.tok2vec = tok2vec
    return model


def block_gradients(model):
    from thinc.api import wrap
    def forward(X, drop=0.):
        Y, _ = model.begin_update(X, drop=drop)
        return Y, None
    return wrap(forward, model)


def create_pipeline(width, embed_size, vectors_model, use_vectors):
    print("Load vectors")
    nlp = spacy.load(vectors_model)
    print("Start training")
    if use_vectors:
        tok2vec = Tok2Vec(width=width, embed_size=embed_size,
                    pretrained_vectors=nlp.vocab.vectors.name)
    else:
        tok2vec = Tok2Vec(width=width, embed_size=embed_size)
    textcat = TextCategorizer(nlp.vocab, 
        labels=['POSITIVE', 'NEGATIVE'],
        model=build_textcat_model(tok2vec, 2, width))
    nlp.add_pipe(textcat)
    return nlp


def train_tensorizer(nlp, texts, dropout, n_iter):
    tensorizer = nlp.create_pipe('tensorizer')
    nlp.add_pipe(tensorizer)
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        losses = {}
        for i, batch in enumerate(minibatch(texts)):
            docs = [nlp.make_doc(text) for text in batch]
            tensorizer.update(docs, None, losses=losses, sgd=optimizer, drop=dropout)
        send_metrics(pretrain_loss=losses['tensorizer'])
        print(losses)
    return optimizer


def train_textcat(nlp, n_texts, opt_params, n_iter=10, dropout=0.2, batch_size=2):
    textcat = nlp.get_pipe('textcat')
    tok2vec_weights = textcat.model.tok2vec.to_bytes()
    (train_texts, train_cats), (dev_texts, dev_cats) = load_textcat_data(limit=n_texts)
    print("Using {} examples ({} training, {} evaluation)"
          .format(n_texts, len(train_texts), len(dev_texts)))
    train_data = list(zip(train_texts,
                          [{'cats': cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    best_acc = 0.0
    early_stopping = EarlyStopping('acc', 5)
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        # Params arent passed in properly in spaCy :(. Work around the bug.
        optimizer = nlp.begin_training()
        configure_optimizer(optimizer, opt_params)
        textcat.model.tok2vec.from_bytes(tok2vec_weights)
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {'textcat': 0.0}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=batch_size)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=dropout,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate_textcat(nlp.tokenizer, textcat, dev_texts, dev_cats)
            best_acc = max(best_acc, scores['acc'])
            report_progress(i, best_acc, losses, scores)
            should_stop = early_stopping.update(scores)
            if should_stop:
                break

def evaluate_textcat(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8
    fp = 1e-8
    tn = 1e-8
    fn = 1e-8
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score,
            'acc': (tp+tn) / (tp+tn+fp+fn)}

def get_opt_params(kwargs):
    return {
        'learn_rate': kwargs['learn_rate'],
        'optimizer_B1': kwargs['b1'],
        'optimizer_B2': kwargs['b1'] * kwargs['b2_ratio'],
        'optimizer_eps': kwargs['adam_eps'],
        'L2': kwargs['L2'],
        'grad_norm_clip': kwargs['grad_norm_clip'],
    }

def configure_optimizer(opt, params):
    # These arent passed in properly in spaCy :(. Work around the bug.
    opt.alpha = params['learn_rate']
    opt.b1 = params['optimizer_B1']
    opt.b2 = params['optimizer_B2']
    opt.eps = params['optimizer_B2']
    opt.L2 = params['L2']
    opt.max_grad_norm  = params['grad_norm_clip']


MAIN_ARGS = {
    'width': ("Width of CNN layers", "positional", None, int),
    'embed_size': ("Embedding rows", "positional", None, int),
    'pretrain_iters': ("Number of iterations to pretrain", "option", "pn", int),
    'train_iters': ("Number of iterations to pretrain", "option", "tn", int),
    'train_examples': ("Number of labelled examples", "option", "eg", int),
    'batch_size': ("Batch_size for TC", "option", "bs", int),
    'dropout': ("Dropout for TC", "option", "do", float),
    'pretrain_dropout': ("Dropout for pretrain", "option", "pd", float),
    'vectors_model': ("Name or path to vectors model to learn from"),
    'use_vectors': ("Whether to use the vectors in the input", "option", 'UV', int),
    'learn_rate': ("Learning rate for TC", "option", "lr", float),
    'b1': ("First momentum term for Adam", "option", "b1", float),
    'b2_ratio': ("Ratio between b1 and b2 for Adam", "option", "b2r", float),
    'adam_eps': ("Adam epsilon", "option", "adam_eps", float),
    'L2': ("L2 penalty", "option", "L2", float),
    'grad_norm_clip': ("Clip gradient by L2 norm", "option", "gc", float)
}

def main(width: int, embed_size: int, vectors_model,
        pretrain_iters=30, train_iters=30, train_examples=1000,
        use_vectors=0, batch_size=2, dropout=0.2, pretrain_dropout=0.2,
        learn_rate=0.001, b1=0.9, b2_ratio=100, adam_eps=1e-12, L2=0.0,
        grad_norm_clip=1.0):
    opt_params = get_opt_params(locals())
    random.seed(0)
    numpy.random.seed(0)
    use_gpu = prefer_gpu()
    print("Using GPU?", use_gpu)

    nlp = create_pipeline(width, embed_size, vectors_model, use_vectors)
    print("Load data")
    texts = load_texts(limit=0)
    print("Train tensorizer")
    optimizer = train_tensorizer(nlp, texts,
                    dropout=pretrain_dropout, n_iter=pretrain_iters)
    print("Train textcat")
    train_textcat(nlp, train_examples, opt_params,
        dropout=dropout, batch_size=batch_size, n_iter=train_iters)

if __name__ == '__main__':
    if 1:
        args = polyaxon_helper.get_declarations()
        print(args)
        main(**args)
    else:
        plac.call(plac.annotations(**MAIN_ARGS)(main))
