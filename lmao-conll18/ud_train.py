# flake8: noqa
"""Train for CONLL 2017 UD treebank evaluation. Takes .conllu files, writes
.conllu format for development data, allowing the official scorer to be used.
"""
from __future__ import unicode_literals

import plac
import tqdm
from pathlib import Path
import re
import sys
import json


import spacy
import spacy.util
from spacy.tokens import Token, Doc
from spacy.gold import GoldParse
from spacy.util import compounding, minibatch, minibatch_by_words
from spacy.syntax.nonproj import projectivize
from spacy.matcher import Matcher
from spacy import displacy
from collections import defaultdict, Counter
from timeit import default_timer as timer

import itertools
import random
import numpy.random

from spacy.cli.ud import conll17_ud_eval

from spacy import lang
from spacy.lang import zh
from spacy.lang import ja

try:
    import torch
except ImportError:
    torch = None


################
# Data reading #
################

space_re = re.compile("\s+")


def split_text(text):
    return [space_re.sub(" ", par.strip()) for par in text.split("\n\n")]


def read_data(
    nlp,
    conllu_file,
    text_file,
    raw_text=True,
    oracle_segments=False,
    max_doc_length=None,
    limit=None,
):
    """Read the CONLLU format into (Doc, GoldParse) tuples. If raw_text=True,
    include Doc objects created using nlp.make_doc and then aligned against
    the gold-standard sequences. If oracle_segments=True, include Doc objects
    created from the gold-standard segments. At least one must be True."""
    if not raw_text and not oracle_segments:
        raise ValueError("At least one of raw_text or oracle_segments must be True")
    paragraphs = split_text(text_file.read())
    conllu = read_conllu(conllu_file)
    # sd is spacy doc; cd is conllu doc
    # cs is conllu sent, ct is conllu token
    docs = []
    golds = []
    for doc_id, (text, cd) in enumerate(zip(paragraphs, conllu)):
        sent_annots = []
        for cs in cd:
            sent = defaultdict(list)
            for id_, word, lemma, pos, tag, morph, head, dep, _, space_after in cs:
                if "." in id_:
                    continue
                if "-" in id_:
                    continue
                id_ = int(id_) - 1
                head = int(head) - 1 if head != "0" else id_
                sent["words"].append(word)
                sent["tags"].append(tag)
                sent["heads"].append(head)
                sent["deps"].append("ROOT" if dep == "root" else dep)
                sent["spaces"].append(space_after == "_")
            sent["entities"] = ["-"] * len(sent["words"])
            sent["heads"], sent["deps"] = projectivize(sent["heads"], sent["deps"])
            if oracle_segments:
                docs.append(Doc(nlp.vocab, words=sent["words"], spaces=sent["spaces"]))
                golds.append(GoldParse(docs[-1], **sent))

            sent_annots.append(sent)
            if raw_text and max_doc_length and len(sent_annots) >= max_doc_length:
                doc, gold = _make_gold(nlp, None, sent_annots)
                sent_annots = []
                docs.append(doc)
                golds.append(gold)
                if limit and len(docs) >= limit:
                    return docs, golds

        if raw_text and sent_annots:
            doc, gold = _make_gold(nlp, None, sent_annots)
            docs.append(doc)
            golds.append(gold)
        if limit and len(docs) >= limit:
            return docs, golds
    return docs, golds


def read_conllu(file_):
    docs = []
    sent = []
    doc = []
    for line in file_:
        if line.startswith("# newdoc"):
            if doc:
                docs.append(doc)
            doc = []
        elif line.startswith("#"):
            continue
        elif not line.strip():
            if sent:
                doc.append(sent)
            sent = []
        else:
            sent.append(list(line.strip().split("\t")))
            if len(sent[-1]) != 10:
                print(repr(line))
                raise ValueError
    if sent:
        doc.append(sent)
    if doc:
        docs.append(doc)
    return docs


def _make_gold(nlp, text, sent_annots, drop_deps=0.0):
    # Flatten the conll annotations, and adjust the head indices
    flat = defaultdict(list)
    sent_starts = []
    for sent in sent_annots:
        flat["heads"].extend(len(flat["words"]) + head for head in sent["heads"])
        for field in ["words", "tags", "deps", "entities", "spaces"]:
            flat[field].extend(sent[field])
        sent_starts.append(True)
        sent_starts.extend([False] * (len(sent["words"]) - 1))
    # Construct text if necessary
    assert len(flat["words"]) == len(flat["spaces"])
    if text is None:
        text = "".join(
            word + " " * space for word, space in zip(flat["words"], flat["spaces"])
        )
    doc = nlp.make_doc(text)
    flat.pop("spaces")
    gold = GoldParse(doc, **flat)
    gold.sent_starts = sent_starts
    for i in range(len(gold.heads)):
        if random.random() < drop_deps:
            gold.heads[i] = None
            gold.labels[i] = None

    return doc, gold


#############################
# Data transforms for spaCy #
#############################


def golds_to_gold_tuples(docs, golds):
    """Get out the annoying 'tuples' format used by begin_training, given the
    GoldParse objects."""
    tuples = []
    for doc, gold in zip(docs, golds):
        text = doc.text
        ids, words, tags, heads, labels, iob = zip(*gold.orig_annot)
        sents = [((ids, words, tags, heads, labels, iob), [])]
        tuples.append((text, sents))
    return tuples


##############
# Evaluation #
##############


def evaluate(nlp, text_loc, gold_loc, sys_loc, limit=None):
    if text_loc.parts[-1].endswith(".conllu"):
        docs = []
        with text_loc.open(encoding='utf8') as file_:
            for conllu_doc in read_conllu(file_):
                for conllu_sent in conllu_doc:
                    words = [line[1] for line in conllu_sent]
                    docs.append(Doc(nlp.vocab, words=words))
        for name, component in nlp.pipeline:
            docs = list(component.pipe(docs))
    else:
        with text_loc.open("r", encoding="utf8") as text_file:
            texts = split_text(text_file.read())
            docs = list(nlp.pipe(texts))
    with sys_loc.open("w", encoding="utf8") as out_file:
        write_conllu(docs, out_file)
    with gold_loc.open("r", encoding="utf8") as gold_file:
        gold_ud = conll17_ud_eval.load_conllu(gold_file)
        with sys_loc.open("r", encoding="utf8") as sys_file:
            sys_ud = conll17_ud_eval.load_conllu(sys_file)
        scores = conll17_ud_eval.evaluate(gold_ud, sys_ud)
    return docs, scores


def write_conllu(docs, file_):
    merger = Matcher(docs[0].vocab)
    merger.add("SUBTOK", None, [{"DEP": "subtok", "op": "+"}])
    for i, doc in enumerate(docs):
        matches = merger(doc)
        spans = [doc[start : end + 1] for _, start, end in matches]
        offsets = [(span.start_char, span.end_char) for span in spans]
        for start_char, end_char in offsets:
            doc.merge(start_char, end_char)
        file_.write("# newdoc id = {i}\n".format(i=i))
        for j, sent in enumerate(doc.sents):
            file_.write("# sent_id = {i}.{j}\n".format(i=i, j=j))
            file_.write("# text = {text}\n".format(text=sent.text))
            for k, token in enumerate(sent):
                if token.head.i > sent[-1].i or token.head.i < sent[0].i:
                    for word in doc[sent[0].i - 10 : sent[0].i]:
                        print(word.i, word.head.i, word.text, word.dep_)
                    for word in sent:
                        print(word.i, word.head.i, word.text, word.dep_)
                    for word in doc[sent[-1].i : sent[-1].i + 10]:
                        print(word.i, word.head.i, word.text, word.dep_)
                    raise ValueError(
                        "Invalid parse: head outside sentence (%s)" % token.text
                    )
                file_.write(token._.get_conllu_lines(k) + "\n")
            file_.write("\n")


def report_progress(itn, losses, ud_scores, prev_best):
    fields = {
        "epoch": itn,
        "dep_loss": losses.get("parser", 0.0),
        "tag_loss": losses.get("tagger", 0.0),
        "words": ud_scores["Words"].f1 * 100,
        "sents": ud_scores["Sentences"].f1 * 100,
        "tags": ud_scores["XPOS"].f1 * 100,
        "uas": ud_scores["UAS"].f1 * 100,
        "las": ud_scores["LAS"].f1 * 100,
    }
    fields['best'] = max(fields['las'], prev_best)
    send_metrics(**fields)
    return fields['best']


def get_token_conllu(token, i):
    if token._.begins_fused:
        n = 1
        while token.nbor(n)._.inside_fused:
            n += 1
        id_ = "%d-%d" % (i, i + n)
        lines = [id_, token.text, "_", "_", "_", "_", "_", "_", "_", "_"]
    else:
        lines = []
    if token.head.i == token.i:
        head = 0
    else:
        head = i + (token.head.i - token.i) + 1
    fields = [
        str(i + 1),
        token.text,
        token.lemma_,
        token.pos_,
        token.tag_,
        "_",
        str(head),
        token.dep_.lower(),
        "_",
        "_",
    ]
    lines.append("\t".join(fields))
    return "\n".join(lines)


Token.set_extension("get_conllu_lines", method=get_token_conllu, force=True)
Token.set_extension("begins_fused", default=False, force=True)
Token.set_extension("inside_fused", default=False, force=True)


##################
# Initialization #
##################


def load_nlp(corpus, config, vectors=None):
    lang = corpus.split("_")[0]
    nlp = spacy.blank(lang)
    if config.vectors:
        if not vectors:
            raise ValueError(
                "config asks for vectors, but no vectors "
                "directory set on command line (use -v)"
            )
        if (Path(vectors) / corpus).exists():
            nlp.vocab.from_disk(Path(vectors) / corpus / "vocab")
    nlp.meta["treebank"] = corpus
    return nlp


def initialize_pipeline(nlp, docs, golds, config, device):
    nlp.add_pipe(nlp.create_pipe("tagger"))
    nlp.add_pipe(nlp.create_pipe("parser"))
    if config.multitask_tag:
        nlp.parser.add_multitask_objective("tag")
    if config.multitask_sent:
        nlp.parser.add_multitask_objective("sent_start")
    for gold in golds:
        for tag in gold.tags:
            if tag is not None:
                nlp.tagger.add_label(tag)
    if torch is not None and device != -1:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    optimizer = nlp.begin_training(
        lambda: golds_to_gold_tuples(docs, golds),
        device=device,
        subword_features=config.subword_features,
        conv_depth=config.conv_depth,
        bilstm_depth=config.bilstm_depth,
    )
    optimizer.alpha = config.learn_rate
    optimizer.b1 = config.adam_b1
    optimizer.b2 = config.adam_b1 * config.adam_b2_factor
    optimizer.b1_decay = config.adam_b1_decay
    optimizer.b2_decay = config.adam_b2_decay
    optimizer.eps = config.adam_eps
    optimizer.L2 = config.L2

    if config.pretrained_tok2vec:
        _load_pretrained_tok2vec(nlp, config.pretrained_tok2vec)
    return optimizer


def _load_pretrained_tok2vec(nlp, loc):
    """Load pre-trained weights for the 'token-to-vector' part of the component
    models, which is typically a CNN. See 'spacy pretrain'. Experimental.
    """
    with Path(loc).open("rb") as file_:
        weights_data = file_.read()
    loaded = []
    for name, component in nlp.pipeline:
        if hasattr(component, "model") and hasattr(component.model, "tok2vec"):
            component.tok2vec.from_bytes(weights_data)
            loaded.append(name)
    return loaded


########################
# Command line helpers #
########################


class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def load(cls, loc):
        with Path(loc).open("r", encoding="utf8") as file_:
            cfg = json.load(file_)
        return cls(**cfg)


class Dataset(object):
    def __init__(self, path, section):
        self.path = path
        self.section = section
        self.conllu = None
        self.text = None
        for file_path in self.path.iterdir():
            name = file_path.parts[-1]
            if section in name and name.endswith("conllu"):
                self.conllu = file_path
            elif section in name and name.endswith("txt"):
                self.text = file_path
        if self.conllu is None:
            msg = "Could not find .txt file in {path} for {section}"
            raise IOError(msg.format(section=section, path=path))
        if self.text is None:
            msg = "Could not find .txt file in {path} for {section}"
        self.lang = self.conllu.parts[-1].split("-")[0].split("_")[0]


class TreebankPaths(object):
    def __init__(self, ud_path, treebank, **cfg):
        self.train = Dataset(ud_path / treebank, "train")
        self.dev = Dataset(ud_path / treebank, "dev")
        self.lang = self.train.lang

MAIN_ARGS = dict(
    ud_dir=("Path to Universal Dependencies corpus", "positional", None, Path),
    corpus=(
        "UD corpus to train and evaluate on, e.g. en, es_ancora, etc",
        "positional",
        None,
        str,
    ),
    parses_dir=("Directory to write the development parses", "positional", None, Path),
    config=("Path to json formatted config file", "option", "C", Path),
    limit=("Size limit", "option", "n", int),
    gpu_device=("Use GPU", "option", "g", int),
    use_oracle_segments=("Use oracle segments", "flag", "G", int),
    vectors_dir=(
        "Path to directory with pre-trained vectors, named e.g. en/",
        "option",
        "v",
        Path,
    ),
    vectors=("Whether to use vectors", "option", "V", int),
    max_doc_length=("Max sents per doc", "option", "m", int),
    multitask_tag=("Whether to use multitask-tag objective", "option", "mt", int),
    multitask_sent=("Whether to use multitask-sent objective", "option", "ms", int),
    multitask_dep=("Whether to use multitask-dep objective", "option", "md", int),
    multitask_vector=("Whether to use multitask-vector objective", "option", "mv", int),
    bilstm_depth=("BiLSTM depth", "option", "lstm", int),
    subword_features=("Whether to use subword features", "option", "sub", int),
    nr_epoch=("Number of training epochs", "option", "i", int),
    min_batch_size=("Minimum batch size", "option", "bs", int),
    max_batch_factor=("Factor on min_batch_size to grow to", "option", "bf", int),
    dropout=("Dropout", "option", "d", float),
    conv_depth=("CNN depth", "option", "cd", int),
    pretrained_tok2vec=("Pretrained tok2vec weights", "option", "t2v", str)
)

def main(
    ud_dir,
    parses_dir,
    corpus,
    config=None,
    limit=0,
    gpu_device=-1,
    vectors_dir=None,
    use_oracle_segments=False,
    vectors=False,
    learn_rate=0.001,
    adam_b1=0.8,
    adam_b2_factor=1.0,
    adam_b1_decay=0.0,
    adam_b2_decay=0.0,
    adam_eps=1e-5,
    L2=1e-6,
    max_lr=0.01,
    lr_ratio=32,
    lr_steps=10000,
    max_doc_length=10,
    multitask_sent=0,
    multitask_tag=0,
    multitask_dep=0,
    multitask_vector=0,
    bilstm_depth=0,
    nr_epoch=30,
    min_batch_size=100,
    max_batch_factor=10.0,
    dropout=0.2,
    conv_depth=4,
    subword_features=1,
    pretrained_tok2vec=None
):
    config = Config(**locals())
    spacy.util.fix_random_seed()
    ud_dir = Path(ud_dir)
    parses_dir = Path(parses_dir)
    paths = TreebankPaths(ud_dir, corpus)
    if not (parses_dir / corpus).exists():
        (parses_dir / corpus).mkdir()
    print("Train and evaluate", corpus, "using lang", paths.lang)
    nlp = load_nlp(paths.lang, config, vectors=vectors_dir)

    docs, golds = read_data(
        nlp,
        paths.train.conllu.open(encoding='utf8'),
        paths.train.text.open(encoding='utf8'),
        max_doc_length=config.max_doc_length,
        limit=limit,
    )

    optimizer = initialize_pipeline(nlp, docs, golds, config, gpu_device)

    batch_sizes = compounding(
        config.min_batch_size,
        config.min_batch_size * config.max_batch_factor,
        1.001
    )
    best = 0.0
    for i in range(config.nr_epoch):
        docs, golds = read_data(
            nlp,
            paths.train.conllu.open(encoding='utf8'),
            paths.train.text.open(encoding='utf8'),
            max_doc_length=config.max_doc_length,
            limit=limit,
            oracle_segments=use_oracle_segments,
            raw_text=not use_oracle_segments,
        )
        Xs = list(zip(docs, golds))
        random.shuffle(Xs)
        batches = minibatch_by_words(Xs, size=batch_sizes)
        losses = {}
        for batch in batches:
            batch_docs, batch_gold = zip(*batch)
            nlp.update(
                batch_docs,
                batch_gold,
                sgd=optimizer,
                drop=config.dropout,
                losses=losses,
            )

        out_path = parses_dir / corpus / "epoch-{i}.conllu".format(i=i)
        with nlp.use_params(optimizer.averages):
            parsed_docs, scores = evaluate(
                nlp, paths.dev.text, paths.dev.conllu, out_path
            )
            best = report_progress(i, losses, scores, best)


if __name__ == "__main__":
    try:
        import polyaxon_helper
        from polyaxon_helper import send_metrics
        use_polyaxon = True
    except ImportError:
        use_polyaxon = False

    if use_polyaxon:
        USE_TQDM = 0
        args = polyaxon_helper.get_declarations()
        print(args)
        main(**args)
    else:
        USE_TQDM = True
        send_metrics = lambda *args, **kwargs: None
        plac.call(plac.annotations(**MAIN_ARGS)(main))
