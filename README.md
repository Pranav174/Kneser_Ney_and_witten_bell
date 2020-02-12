# Kneser Ney & Witten Bell smoothing
## How to run
```python3
pip install spacy
python language_model.py <n(1-3)> <model[w,k]> <location_to_corpus> 
```
## Details
+ The model first gengerates unigrams, bigrams and trigrams from the input corpus
+ Using either Witten Bell smoothing or Kneser Ney smoothing, the prograbability of a sentence is calculated based on the chain rule on [1,2,3]-gram
+ To deal with out of vacabulary words, a \<unk\> word has been used to replace the top 0.5% list of words ordered by their occurance count
+ \<start\> token has also been appened to the start of the sentence based on size of the n-gram
+ For unigram calculation of witten-bell, as there exists no lower model to backoff to, the function will return unigram maximum likelyhood probability
+ Both witten-bell and kneser ney functions are recursive in nature
+ A discount value of 0.75 was chosen for the Kneser-Ney model

## Observations

Some observed probabilities for 3-gram Witten-Bell smoothing
```
for ['shut', 'not', 'your'] : 0.6684373136403442
for ['i', 'am', 'a'] : 0.028386217245508834
for ['<unk>', '<unk>', '<unk>'] : 0.015536017065742628
for ['shut', 'not', 'why'] : 0.0010601300145292442
```

Some observed probabilities for 3-gram Kneser-Ney smoothing
```
for ['shut', 'not', 'your'] : 0.6262269814748344
for ['i', 'am', 'a'] : 0.0311169755823837
for ['<unk>', '<unk>', '<unk>'] : 0.000976893490603514
for ['shut', 'not', 'why'] : 0.00014742638663298163
```

The models performed similary for likely trigrams, like "shut not your" (a phrase in the corpus) and "i am a" (not in the corpus), as expected as both the models use iterpolation to better estimate the probablity. 

The major difference came with unlikely trigrams like "shut not why" and "\<unk\> \<unk\> \<unk\>"(trigram of only unknown words) where Witten-Bell assigned almost ten times the probability assigned by Kneser-Ney. It is likely due to the standard absolute discount and continuation count for lower order n-grams in Kneser_Ney model.