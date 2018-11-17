# Using Language Models with Approximate Outputs to pre-train spaCy

**NB: This describes work in progress. It reflects expected results, not
necessarily things that are already true.**

This repository contains experiments on spaCy's new "pretrain" command, which
uses a ULMFit/Elmo/BERT/etc-like process for pre-training. We employ a novel
trick which we term Language Modelling with Approximate Outputs (LMAO).
Normally language models have to softmax (or hierarchical softmax) over the
output vocabulary, which requires large hidden layers. We want to train small
hidden layers, as we want spaCy to be cost-effective to run.

Instead of outputting numeric IDs, we predict points in a vector space that's
been pre-trained using an algorithm like GloVe or FastText. This lets the model
learn large target vocabularies, without requiring extra parameters.

## Quickstart

See the `Polyaxon` section below to really work on the experiments --- but you
can also try this out on your laptop. It should run fine without GPU.

### Ubuntu 18.04lts

```bash

cd lmao-imdb-1k
./polyaxon_setup.sh
source env3.6/bin/activate
python pretrain_textcat.py 128 1000 en_vectors_web_lg

```

### Other Linux-like

First, ensure you have Python 3.6 with virtualenv, and a working compiler
toolchain. Then:

```bash

cd lmao-imdb-1k
python3 -m venv env3.6
source env3.6/bin/activate
pip install -U pip
pip install "spacy-nightly==2.1.0a2" polyaxon-helper
python -m spacy download en_vectors_web_lg
python pretrain_textcat.py 128 1000 en_vectors_web_lg
```

## Polyaxon

The code in this repository is written to be run using [Polyaxon](https://polyaxon.com/).
Polyaxon is an open-source framework for executing and managing experiments
using Kubernetes. We use Polyaxon so that we can run a large number of configurations,
to properly explore the hyper-parameter space. We currently do this using random
search, but will try more powerful techniques such as Bayesian optimisation in
future.

Bootstrap scripts to get you started with Polyaxon using Google Kubernetes Engine
can be found in the `terraform-gke` directory. The scripts use Hashicorp Terraform
to make the setup very easy --- have a look!

Once you have your cluster launched and the `create-polyaxon-project` script installed
on your path, run:

```bash

create-polyaxon-project lmao-imdb-1k
cd lmao-imdb-1k
source login.sh
polyaxon run -u
```

This launches an experiment group that runs many experiments in parallel, each
with a different hyper-parameter configuration. You can change what's being run
by editing the `polyaxonfile.yml`, and monitor the progress either in the web
dashboard, or on the commandline. See the Polyaxon docs for details.

## Model

The text classification uses a convolutional neural network to extract context-sensitive vectors for each token in the document. These vectors are then averaged, and the resulting vector is used as input to a softmaxed affine layer. The model can be understood as having the following components:

* Extract (`spacy.tokens.doc.Doc` --> `numpy.ndarray[dtype='uint64']): For each
  token, extract the following numeric IDs: the lower-case word form, length
  1 prefix, length 3 suffix, word shape, and the row of the word's vector in
  a precomputed static embedding table.
* Embed: Separately embed the lower-case form, prefix, suffix and word shape
  IDs, using [hash embeddings](https://support.prodi.gy/t/can-you-explain-how-exactly-hashembed-works/564/2). Concatenate these vectors, along with
  the word's static pre-trained vector. Pass the concatenated vector into
  a maxout layer, and apply layer normalization to the output.
* Encode: A series of convolutional neural network layers are applied to the
  vectors. Each layer is constructed as follows. To compute the output for the word i,
  we concatenate the inputs at i-1, i and i+1. The concatenated vector is then
  passed through a maxout layer, with layer normalization used. A residual
  connection is then applied, i.e. the layer's input is added to its output.
  The output of each layer is used as the input to the next layer.
* Reduce: Average pooling is applied over the output of the CNN, so that
  a single vector is produced representing all of the tokens in the document.
* Predict: A softmax layer is used to predict two class labels, one for
  positive and one for negative.

The text-classification model is created within the `pretrain_textcat.py` file,
mostly using code that can mostly be found in the `Tok2Vec` function in the
`spacy._ml` module.

The model architecture is designed to focus primarily on created
context-sensitive representations of the tokens, which is why the output layer
is kept so simple. We then try to pre-train these context-sensitive
representations, using LMAO.

The pre-training works as follows. First, we compute an embedding matrix for
the document, using the pre-trained embeddings. Next, we predict the vectors
for the document, using the CNN. We then update the CNN, using the mean-squared
error of the distance between the CNN output and the pre-trained embeddings.
Specifically, the gradient of the loss will be:

```python

prediction = CNN(doc)
target = pretrained_embeddings(doc)
d_loss = (prediction - target) / len(doc)
```

Note that the static embeddings are currently provided to the CNN as input, so
in theory we only need to do compression. The model should be able to ignore
the other features, and just learn to use the static embedding. In practice the
dropout does complicate this --- on some percentage of tokens, the static
embedding will be missing, forcing the model to fill in the information with
other features. Still, using the static embedding in the input seems unideal,
but currently results are much worse without it. This puzzle might be resolved
by simply using more data to pre-train the model.


## Experiment

As a first test of LMAO, we're working from the low-data IMDB experiment
performed by Howard and Ruder (2018), specifically Figure 3 of their ULMFit paper.
One of the most striking claims in their paper is that error rates close to the
current state-of-the-art could be achieved using only a fraction of the
labelled training data, by taking advantage of unsupervised pre-training. While
Howard and Ruder report from 100 examples on, we find that training with such
low data is quite volatile, and it's very little effort to collect a few more
examples. We therefore compare on the 1000 sample configuration.

We shuffle the IMDB training data and take the first 1000 examples as trainign
data, and the last 10000 as development data, to evaluate the hyper-parameter
configurations. The test labels are reserved for future final comparison.

We pre-train the CNN using the LMAO objective on all 50,000 IMDB texts (note we
use the test *text* for this, but not the test labels), approximately 10m
words. We pre-train for either 0, 1, 2, 3 or 4 iterations. The pre-trained CNN
weights are then used to initialize the text classification model, which is
trained on the 1000 labelled training examples. 

## Results

We ran 100 configurations of the experiment, using randomly selected
hyper-parameters. To compare the effect of some setting on accuracy, we can
look at the top-3 accuracies produced for each value of that setting.


The main question we want to answer is how the pre-training iterations effect
the model's text classification accuracy. We investigate pre-training for 0, 1,
2 or 3 epochs. We performed a separate hyper-parameter search for each
configuration, performing 50 runs with randomly sampled hyper-parameters each
time (producing 200 runs in total). We report the best accuracy found for each
configuration:

| LMAO?    | Accuracy |
| -------- | -------- |
| Baseline |          | 
| 1 epoch  |          | 
| 2 epoch  |          | 
| 3 epoch  |          | 

The pre-training produces a nice (albeit not earth-shattering) improvement in
accuracy over the baseline. Most of the benefit comes from the first epoch of
pre-training, with subsequent epochs producing little improvement. This likely
indicates that the LMAO objective isn't difficult enough currently, and results
could be improved with further tweaking. 

Full results will be made available as a CSV.  Notably, the full results show
that the same ordering applies if you look at more results --- it's not simply an
outcome of which configuration gets "lucky" at the random search.


## Future work

### Multi-task learning instead of transfer learning?

Maybe we should be interleaving updates for both objectives? It's less
convenient than pre-training, but the LMAO objective is pretty fast, and it
might get much better results.

### More difficult objectives?

Instead of the word's vector, what if we ran the parser, and tried to predict
the vector of its head? We could also concatenate some words from the context,
like trying to predict a trigram instead of the unigram.

### LMFAO?

If we want to apply the LMAO trick to real language modelling, and actually
output IDs, an extra nearest-neighbours computation will be required. This will
probably be expensive if we're using a really large vocabulary. Fortunately
an obvious extension can address this: Language Modelling with Faster
Approximate Outputs would simply use a nearest neighbours technique.
(Suggestions on which library will perform best for k=1 welcome! Currently
Annoy seems like the first thing to try, as it seems to strike a good balance
between being production quality software and performance.)

