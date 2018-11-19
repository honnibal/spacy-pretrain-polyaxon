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

### Experiment 1: IMDB-1k, GloVe vectors

The main question we want to answer is how the pre-training iterations effect
the model's text classification accuracy. We investigate pre-training for 0, 1,
2 or 3 epochs. We performed a separate hyper-parameter search for each
configuration, performing 50 runs with randomly sampled hyper-parameters each
time (producing 200 runs in total). We report the best accuracy found for each
configuration:

| LMAO?    | Accuracy |
| -------- | -------- |
| Baseline | 86.8     | 
| 1 epoch  | 87.5     | 
| 2 epoch  | 86.5     | 
| 3 epoch  | 86.5     | 

This is basically a negative result. We see a small improvement in accuracy
after 1 epoch, but nothing substantial. The accuracy of the LMAO-1-epoch
configuration is up around what I saw in my preliminary experiments, but the
baseline then was around 85%, where now the baseline is coming in at 86.8. Go
figure. The only real difference is that in my preliminary experiment,
I accidentally ran everything against the test set, which obviously isn't
suitable for all this hyper-parameter search. So, maybe just luck of the draw
on the random search?

Eyeballing an ordering of the experiments, the LMAO-1-epoch runs definitely
cluster near the top. So, it does help. A bit.

The real problem here is that the accuracy goes down as more epochs are
trained. To me this indicates the supervision task is too easy: the model can learn
exploits that let it satisfy the loss, without learning more useful vectors.

I think a likely problem is the use of the same vectors on the input as we have
in the output. I think the model can learn to rely on those vectors, and
steadily ignore the context information. To fix this, we can either remove the
vectors from the input, and not have the pre-trained vectors, or we can use
different vectors.

### Experiment 2: OntoNotes NER, FastText vectors

After experiment 1, it'd be good to get a different perspective on how this
works. One thing we want to try is pre-training with much more text. We also
want to pre-train one of the other components.

We first perform a quick test, using only one hyper-parameter configuration ---
the current default's for spaCy's NER. I piped text from the January 2017
partition of the Reddit Comments Corpus through the `spacy pretrain` command,
and saved out the CNN weights every 1 million words. The CNN weights are only
4mb with the width (128) and embedding rows (1000) being investigated, so it's
no problem to try lots of check points.

I set the batch size for pre-training to 128 comments, resulting in a training
speed of 70k words per second on a Tesla v100 GPU -- about a 7-fold speed-up
over the CPU execution speed. Because the model is so narrow, the pre-training
only utilises about 15% of the card, leaving plenty of capacity to also run the
NER experiments on the same device. 

For this experiment, I used the FastText common-crawl vectors with subword
features as the objective, instead of the GloVe common crawl model used in
experiment 1. While the `spacy pretrain` command dumped out the vector
checkpoints, I periodically started NER models training, using CNN models
pre-trained on increasing amounts of data.

Here's how F-measure on the development data looked for each epoch of training,
for the different models I trained. The number is the amount of data the
pre-training ran over (in a single pass --- the text is effectively infinite,
so why train on the same text twice?). The T indicates a trigram objective,
i.e. the output is a trigram of vectors. I started trying this because
I thought the unigram objective might be too easy.

| Epoch | Baseline | 50**7 | 10**8`| T 10**7 | T 50**7  | T 10**8 |
| ----- | -------- | ----- | ----- | ------- | -------- | ------- |
| 1     | 78.3     | 80.7  | 80.8  | 80.4    | 80.4     | 80.2    |
| 2     | 81.4     | 82.6  | 82.9  | 82.5    | 82.7     | 82.7
| 3     | 82.6     | 83.7  | 83.9  | 83.5    | 83.9     | 83.7
| 4     | 83.5     | 84.4  | 84.5  | 84.1    | 84.5     | 84.3
| 5     | 83.7     | 84.5  | 84.8  | 84.3    | 84.6     | 84.8
| 6     | 83.9     | 85.0  | 85.2  | 84.3    | 84.9     | 85.0
| 7     | 84.2     | 85.1  | 85.2  | 84.6    | 85.1     | 85.0
| 8     | 84.4     | 85.1  | 85.3  | 84.7    | 85.1     | 85.
| 9     | 84.7     | 85.1  | 85.5  | 84.7    | 85.2     | 85. 
| 10    | 84.7     | 85.2  | 85.6  | (stop)  | 85.4     | 85.
| 11    | 84.7     | 85.3  | 85.7  |         | 85.3     | 85.
| 12    | 84.9     | 85.3  | 85.5  |         | 85.3     | 85.
| 13    | 85.0     | 85.4  | 85.6  |         | 85.5     | 
| 14    | 85.1     | 85.4  | 85.5  |         | 85.6     | 
| 15    | 85.2     | 85.5  | 85.8  |         | 85.5     |
| 16    | 85.2     | 85.4  | 85.8  |         | 85.5     |
| 17    |          | 85.6  | 85.7  |         | 85.5     |
| Best  | 85.5     | 85.6  | 85.8  |         | 85.7     | 85.7  

So this hasn't really worked. There's an improvement towards the start of
training, but eventually the baseline overcomes its worse initialization, and
converges to the same sort of score.

The following explanations seem most likely to me (in order).

1. The dataset is big enough to train a model of this size. If the dataset is
   big enough for the model, pre-training doesn't help.
2. The objective isn't good enough. Semi-supervised learning really needs to be
   *prediction*, not mere compression. This result from the auto-encoding
   literature is apparently very well supported, although the logic of it is
   a mystery to me, and I don't have the citations at hand.
3. Everything's fine, we just need more of the same. More training, tweaked
   hyper-parameters, maybe a small trick here or there, etc.
4. Pre-training the CNN doesn't work, for some reason. BiLSTM and the
   transformer have important differences, especially about long-range
   dependencies.


I think 1 is probably the most conventional logic of pre-training. Certainly
the authors of the BERT paper make a comment like this, saying "Everything
we've observed suggests bigger models are better". This is kind of a bummer
though! Even though this is likely to be true, I want to investigate carefully
to make sure of it --- because if it isn't, we can have both efficiency and
accuracy.

The next experiments are therefore to change the objective. One idea I've had
for a long time is to use the parser for this purpose. Instead of the word's
head or a trigram of the word and its surrounding context, we can predict the
vector of the word's syntactic head (using the parser's prediction). I've tried
making a bigram of the word's head and the word itself. It seems good that the
word should keep the knowledge of what itself is, although maybe that's
unnecessary.

While I didn't bother to log the results, the trigram and parser objectives
didn't significantly change the story. The accuracy from the first iterations
was about the same as the other techniques, or even worse. It doesn't make
sense to me that the pre-training could make the accuracy *worse* at first, and
later better. The more epochs we do, the further we go away from the
initialisation. If the initialisation is worse at first, it could only be luck
that carries us to a better result. So, I would say an early lead should be
necessary but not sufficient for success here.

I then left it to pretrain with the original (single token) objective,
a wider model (300 dimensions, up from 128), and more embedding rows (5000).
If the problem is that the objective is too easy, making the model larger
and leaving it to pretrain for longer shouldn't work. If the problem is that
the model is too small and it's already "saturated" by the training data, the
results should be better here. Of course a null result is hard to interpret:
there's no limit to the number of ways an idea can fail to work, and multiple
problems can occur at once.

Task: NER
Objective: Focus word's FastText vector
Width: 300
Number of embedding rows: 5000

| Epoch | Baseline | 10**8 | 20**9 | P 10**7 |
| ----- | -------- | ----- | ----- | ------- |
| 0 | 80.6 | 82.8 | 82.7 |
| 1 | 82.9 | 84.4 | 84.1 |
| 2 | 83.7 | 84.9 | 84.7 |
| 3 | 84.4 | 85.3 | 85.0 | 
| 4 | 84.6 | 85.4 | 85.0 | 
| 5 | 84.9 | 85.5 | 85.5 | 
| 6 | 85.1 | 85.7 | 85.6 |
| 7 | 85.2 | 85.6 | 85.7 | 
| 8 | 85.1 | 85.8 | 85.8 | 
| 9 | 85.3 | 85.8 | 85.8 | 
| 10 | 85.4 | 85.7 | 85.7 |
| 11 | 85.4 | 85.6 | 85.7 | 
| 12 | 85.3 | 85.5 | 85.7 | 
| 13 | 85.3 | 85.6 | 85.8 | 
| 14 | 85.2 | 85.6 | 85.6 | 
| 15 | 85.2 | 85.5 | 85.7 | 

The results here are pretty much as before, so not much has changed simply from
making the model larger. The baseline gets slightly worse, while the accuracy
of the pre-trained models stays solid at 85.8. The differences are pretty
small. Using weights pre-training on 10**8 words of text does just as well as
weights pre-trained on 20**9 words. Again, it's hard to interpret negative
results, but these are at least consistent with the idea that the pre-training
objective is too easy.

I'll try the parser-based objective again, but modify it so that it's only
predicting the word's head, rather than a concatenation of the word's vector
and it's head. Maybe that's too difficult, or otherwise confusing for the
model.

Results with width 128, embed rows 1000. 10**7: 80.047, 82.4, 83.5 (stopped).
20**7: 80.07, 82.3 (stop), 38**7: 79.9 (stopped).

Okay, so what if it's explanation 3: fiddly details? The first thing to try is
to prevent the forgetting that's likely to occur. After all, we shouldn't
expect initialization to matter so much --- we're asking the optimizer to
be insensitive to that. We can try Howard and Ruder's unfreezing, but the easy
to implement first thing is to just dampen the learning rate on the CNN. It's
also worth trying to disable the dropout, so that the model learns in fewer
epochs. This seems more promising. With width 300, embed rows 5000, 10**8:
83.8, 85.0, 85.3, 85.8, 85.6. Maybe removing the dropout is bad. 82.7, 83.9,
84.6, 85.2, 85.3, 85.5, 85.8, 85.7, 85.6. So, no real effect from damping the
learning rate. I tried a few different things, half, a tenth, freezing for the
first epoch, etc. Staying frozen for too long harmed final accuracy, everything
else didn't matter.

## LSTM?

What if we tried with an LSTM, so we can use a predict-next-word objective?

Trying with width=128, depth=2, embed_rows=1000, FastText vectors. Immediately
I'm having trouble getting the LSTM to converge. After 50m words the loss-per-batch
isn't really decreasing. I'll try removing the dropout to help it a little. Nope.
Okay, I'll try fitting a small dataset, say 1k comments (about 300k words). Yep,
learns.  100k comments (30m words)?  On the fourth epoch now, and the loss does
seem to be going down. I guess the streaming case is very difficult, so it
might not be problematic if the loss is very slow to decrease. It would be
interesting if this were the problem, though. I would've thought generalisation
would be better from N steps on entirely novel data, instead of using multiple
iterations over the same data?

Currently the LM initialization is causing much worse performance. On the first
epoch, we get to 75 F without, 70 with. Maybe it's not being loaded correctly.
I'll focus on getting the pretraining to fit first.

It seems to help to add a non-linear layer between the LSTM and the output.
Widening to 300 also makes the model fit a lot better. I've added a length cap
to the inputs of 200 tokens, so that the model can be widened without memory
limitations.

Okay, so there was indeed a problem with saving and loading the LSTM weights!
Having fixed this, the pre-training indeed seems to be helping. With width 128
and 2 layers, we get 75 F without pre-training in the first epoch, and having
pre-trained for 60m words, we get 76.9. I also did a quick check when little
training data was available (1k texts), and the pre-training was helping there
as well.

Now time to write up the bidirectional model, as the forward one is getting
a bit worse results than the CNN, and trying only forward is a bit artificial.

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

