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

## 02-12-18: Good results for small models, low data

The `spacy pretrain` command now uses a BERT-style masked language model, but
instead of predicting the word IDs, we predict the GloVe vector of the target
words. I've used this objective to pre-train spaCy's small CNN model on
2 billion words of text from Reddit (the January 2017 portion of the Reddit
comments corpus). The pre-trained CNN is very small: it's depth 4, width 96,
and has only 2000 rows in the hash embeddings table. Weights for the serialized
model are only 3.2 MB.

Here's how pretraining affects parser accuracy on different sized subsets of
the training data.

| #Docs | #Words | Baseline | Pretrain |
| ----- | ------ | -------- | -------- |
| 1     | 69     | 41.8     | 60.5     |
| 10    | 311    | 47.4     | 66.4     |
| 100   | 2947   | 61.3     | 79.0     |
| 1000  | 31097  | 78.5     | 84.7     |
| 10000 | 255000 | 87.0     | 87.8     |

The effect size here is pretty strong. The pretraining is very effective in the
low and medium data cases, but the impact fades as more training data becomes
available. One caveat with these results is that static vectors weren't used in
the input --- the vectors were trained from scratch. This is a bit unfair to
the baseline, since pre-trained vectors are a simple and effective existing way
to get some pretrained knowledge into the model. As a quick datapoint, the result
for baseline+GloVe vectors for the 1000 document case is 82.0%. So the static vectors
aren't as good as the pretraining. We can also use static vectors
together with pretraining, but I need to rerun the pretraining for that.

Low and medium data use-cases are very important, because they help you perform
rapid prototyping. When doing practical NLP, there are an enormous number of
decisions to make in how you break down the application requirements into
statistical models. To make good decisions, you need to try things out and see
how well models perform with different approaches. If you have models which
take huge quantities of data before they yield reasonable performance, you have
to do a lot of annotation just to find out an idea doesn't work.

A reasonable interpretation of the current results is that pretraining lets you
use a larger model than your training data could otherwise support. The
previous work has shown that pretraining lets us make very large models, and
those models will let us hit state-of-the-art accuracies even when the training
corpus is quite large. It wasn't clear from the previous work whether
pretraining would also improve results for a small model. My results so far
suggest it does, but so far only when the training data is small.

I'm now pre-training a larger CNN, to check that my results match up with
previous work in finding that this improves accuracy on the full dataset. This
could also be a nice option for spaCy users, who want more accuracy even if it
means slower runtimes.


