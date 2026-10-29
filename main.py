# Import necessary libraries
from gensim.test.utils import datapath
import gensim.downloader as api
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
# Load pretrained word embeddings
glove = api.load("glove-twitter-200")
fasttext = api.load("fasttext-wiki-news-subwords-300")
print("GloVe vocabulary size:", len(glove.key_to_index))
print("FastText vocabulary size:", len(fasttext.key_to_index))
# Path to your CSV dataset
wordsim_path = r"C:\Users\HP\Desktop\New ML\Glove-fasttext\wordsim353\combined.csv"
# Convert CSV to a tab-separated file for Gensim
df = pd.read_csv(wordsim_path)
print("\nDataset preview:")
print(df.head())
print("\nColumn names:", df.columns.tolist())
# Rename columns to the format expected by Gensim
df.columns = ["Word 1", "Word 2", "Human judgment"]
# Save as TSV
tsv_path = r"C:\Users\HP\Desktop\New ML\Glove-fasttext\wordsim353.tsv"
df.to_csv(tsv_path, sep="\t", index=False, header=False)
# Evaluate both embedding models
print("\nEvaluation on WordSim-353 dataset")
glove_wordsim = glove.evaluate_word_pairs(datapath(tsv_path))
fasttext_wordsim = fasttext.evaluate_word_pairs(datapath(tsv_path))
# # GloVe results
print("\nGloVe Results")
print("Pearson correlation:", glove_wordsim[0][0])
print("Spearman correlation:", glove_wordsim[1].correlation)
print("OOV ratio:", glove_wordsim[2])
# # FastText results
print("\nFastText Results")
print("Pearson correlation:", fasttext_wordsim[0][0])
print("Spearman correlation:", fasttext_wordsim[1].correlation)
print("OOV ratio:", fasttext_wordsim[2])

test_words  = ['king','queen','apple','apples','run','running','happyness','data','awesome','covid19']
for word in test_words:
    print(f"\nWord : {word}")
    if word in glove:
        glove_neighbors = [w for w,_ in glove.most_similar(word,topn=5)]
        print("GloVe: ",glove_neighbors)
    else:
        print("GloVe: OOV(not in vocabulary)")
    if word in fasttext:
        fasttext_neighbors = [w for w,_ in fasttext.most_similar(word,topn=5)]
    else:
        print("Fast text : OOV(not in vocabulary)")