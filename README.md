# GloVe vs FastText: Word Embedding Evaluation

## Overview

This project compares two pretrained word embedding models, **GloVe** and **FastText**, to explore how they represent semantic relationships between words.

The project evaluates both models using the **WordSim-353 dataset**, which contains human-rated word pairs, and investigates their nearest neighbors for selected words. It also examines vocabulary coverage and how FastText handles words that are not explicitly present in its vocabulary.

The goal is to understand the differences between traditional word embeddings and subword-based representations through practical NLP experimentation.

## Objectives

- Load and explore pretrained GloVe and FastText embeddings.
- Evaluate word similarity using the WordSim-353 dataset.
- Compare model predictions with human similarity judgments.
- Measure Pearson and Spearman correlation.
- Investigate vocabulary coverage and out-of-vocabulary (OOV) behavior.
- Explore semantic relationships through nearest-neighbor analysis.

## Models Used

### 1. GloVe

GloVe (Global Vectors for Word Representation) learns word vectors from global word co-occurrence statistics.

- **Pretrained model:** `glove-twitter-200`
- **Vector dimensions:** 200
- **Training corpus:** Twitter text

### 2. FastText

FastText represents words using character-level subword information, allowing it to construct vectors for some words that are not explicitly stored in its vocabulary.

- **Pretrained model:** `fasttext-wiki-news-subwords-300`
- **Vector dimensions:** 300
- **Training corpus:** Wikipedia and news text

## Dataset

**WordSim-353** is a benchmark dataset containing 353 word pairs with human-assigned similarity ratings.

The dataset is used to evaluate how closely the similarity scores produced by each model correspond to human judgments.

Dataset source: [WordSim-353](https://gabrilovich.com/resources/data/wordsim353/wordsim353.html)

The dataset is prepared as a tab-separated file for compatibility with Gensim's word-pair evaluation functionality.

## Methodology

### 1. Load pretrained embeddings

Both pretrained models are loaded using Gensim's downloader.

### 2. Prepare the dataset

The WordSim-353 CSV file is read using pandas, its columns are checked, and the data is converted to a tab-separated format.

### 3. Evaluate word similarity

Both models are evaluated against the human similarity ratings using Gensim's `evaluate_word_pairs()` method.

The following metrics are examined:

- **Pearson correlation:** Measures the linear relationship between model-generated similarity scores and human ratings.
- **Spearman correlation:** Measures agreement between the rankings of model-generated scores and human judgments.
- **OOV ratio:** Measures the proportion of word pairs that cannot be evaluated because of missing vocabulary.

### 4. Nearest-neighbor analysis

The models are tested using words such as:

`king`, `queen`, `apple`, `apples`, `run`, `running`, `happyness`, `data`, `awesome`, and `covid19`.

The `most_similar()` method retrieves the nearest word vectors using cosine similarity.

### 5. Vocabulary and OOV investigation

The project compares explicit vocabulary membership and investigates FastText's ability to generate representations using subword information.

## Technologies Used

- Python
- Gensim
- Pandas
- NumPy
- SciPy
- Word embeddings
- Natural Language Processing (NLP)
- Cosine similarity
- Pearson and Spearman correlation

## Project Structure

```text
Glove-fasttext/
├── wordsim353/
│   └── combined.csv
├── main.py
├── README.md
└── requirements.txt
```

The filenames are illustrative; adjust them to match your actual repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/Vedline2547/Glove-fasttext.git
cd Glove-fasttext
```

Install the dependencies:

```bash
pip install gensim pandas numpy scipy
```

Run the Python script:

```bash
python main.py
```

The first run may download the pretrained embedding models, which can require substantial disk space and time.

## Results and Analysis

The evaluation produces Pearson correlation, Spearman correlation, and OOV ratio for both models.

The nearest-neighbor experiment provides additional insight into the words associated with selected terms.

Actual evaluation scores and conclusions should be added after running the experiments.

## Key Learnings

- Understanding pretrained word embeddings and their applications in NLP.
- Comparing word representations learned using different approaches.
- Evaluating semantic similarity using human-rated benchmark data.
- Interpreting Pearson and Spearman correlation.
- Exploring cosine similarity and nearest-neighbor retrieval.
- Investigating vocabulary coverage and subword-based OOV handling.
- Building reproducible NLP evaluation workflows using Python.

## Future Improvements

- Visualize embedding relationships using PCA or t-SNE.
- Create a comparison table or chart of evaluation metrics.
- Test additional pretrained embedding models.
- Investigate the effects of misspellings and morphological variations.
- Explore word analogy tasks and other semantic benchmarks.
- Evaluate the models on domain-specific vocabulary.

## Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer

GitHub: [Vedline2547](https://github.com/Vedline2547)

LinkedIn: [Vedline Ochieng](https://www.linkedin.com/in/vedline-ochieng-025146290/)

---

*This project is an educational exploration of pretrained word embeddings and comparative NLP evaluation.*
