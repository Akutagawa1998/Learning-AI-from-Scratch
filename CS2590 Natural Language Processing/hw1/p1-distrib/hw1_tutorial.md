# HW1 Part 1: Logistic Regression — Function-by-Function Pseudocode

This tutorial gives **detailed pseudocode templates** for each required function in Part 1. It does **not** include final code; it tells you what to implement and where.

Target file: `CS2590 Natural Language Processing/hw1/p1-distrib/models.py`

---

## 1) `UnigramFeatureExtractor.__init__(self, indexer)`

**Goal:** store the `Indexer` and any configuration.

**Pseudocode**
```
function __init__(indexer):
    self.indexer = indexer
    # optional config flags
    # self.use_counts = True
    # self.add_bias = True
```

**Notes**
- `indexer` maps feature strings to indices.
- You can add options (counts vs binary, add bias) if you want.

---

## 2) `UnigramFeatureExtractor.get_indexer(self)`

**Goal:** return the indexer so the rest of the code can see feature dimensionality.

**Pseudocode**
```
function get_indexer():
    return self.indexer
```

---

## 3) `UnigramFeatureExtractor.extract_features(self, sentence, add_to_indexer=False)`

**Goal:** map a list of tokens to a sparse feature vector (Counter of index -> value).

**Pseudocode**
```
function extract_features(sentence, add_to_indexer=False):
    feats = Counter()

    # optional bias feature
    # if self.add_bias:
    #     bias_idx = self.indexer.add_and_get_index("BIAS", add_to_indexer)
    #     if bias_idx != -1:
    #         feats[bias_idx] += 1

    for word in sentence:
        # choose how to represent a unigram feature name
        # simplest: use the raw word as the feature
        feat_name = word

        idx = self.indexer.add_and_get_index(feat_name, add_to_indexer)
        if idx != -1:
            # counts
            feats[idx] += 1
            # if you want binary instead:
            # feats[idx] = 1

    return feats
```

**Notes**
- `sentence` is already lowercased in `sentiment_data.py`.
- `add_to_indexer=False` at test time; unseen features are ignored.

---

## 4) `LogisticRegressionClassifier.__init__(self, weights, feat_extractor)`

**Goal:** store trained weights and the feature extractor for inference.

**Pseudocode**
```
function __init__(weights, feat_extractor):
    self.weights = weights
    self.feat_extractor = feat_extractor
```

**Notes**
- `weights` should be a 1-D numpy array of size = number of features.

---

## 5) `LogisticRegressionClassifier.predict(self, ex_words)`

**Goal:** compute probability and output label 0/1.

**Pseudocode**
```
function predict(ex_words):
    feats = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)

    score = 0.0
    for (idx, value) in feats.items():
        score += self.weights[idx] * value

    # sigmoid
    # optional: score = clip(score, -20, 20)
    prob = 1.0 / (1.0 + exp(-score))

    if prob >= 0.5:
        return 1
    else:
        return 0
```

**Notes**
- Using 0.5 as threshold is standard for binary LR.

---

## 6) `train_logistic_regression(train_exs, feat_extractor)`

**Goal:** train weights with SGD on logistic regression.

### Step A: build the feature space
**Pseudocode**
```
for ex in train_exs:
    feat_extractor.extract_features(ex.words, add_to_indexer=True)
```

### Step B: initialize weights
**Pseudocode**
```
num_feats = len(feat_extractor.get_indexer())
weights = zeros(num_feats)
```

### Step C: training loop (SGD)
**Pseudocode**
```
num_epochs = 10   # you can tune
lr = 0.1          # you can tune

for epoch in range(num_epochs):
    shuffle(train_exs)

    for ex in train_exs:
        feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)

        # compute score
        score = 0.0
        for (idx, value) in feats.items():
            score += weights[idx] * value

        # sigmoid
        prob = 1.0 / (1.0 + exp(-score))
        y = ex.label  # 0 or 1

        # gradient for log-loss: (prob - y) * x
        for (idx, value) in feats.items():
            grad = (prob - y) * value
            weights[idx] -= lr * grad

    # optional: compute training loss / dev accuracy for debugging
```

### Step D: return classifier
**Pseudocode**
```
return LogisticRegressionClassifier(weights, feat_extractor)
```

**Notes**
- Shuffling each epoch is important.
- You can decay `lr` by epoch if you want (not required).
- Keep everything sparse using Counter to stay fast.

---

## Quick checklist
- [ ] UnigramFeatureExtractor stores indexer and returns Counter
- [ ] LogisticRegressionClassifier stores weights + extractor
- [ ] Training builds vocab, initializes weights, runs SGD
- [ ] Predict uses sigmoid + threshold

---

## Verify (command)
Run in `p1-distrib`:
```
python sentiment_classifier.py --model LR --feats UNIGRAM
```
Expected: dev accuracy ≥ 0.77 and runtime < 20 seconds.
