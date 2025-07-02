

import nltk
import spacy
from pathlib import Path
import re
import pandas as pd 
import pickle
import os
import math
from collections import Counter
from pathlib import Path


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    sentence_count = 0
    word_count = 0
    syllable_count = 0

    for t in text:
        if t.text in [".", "!", "?"]:
            sentence_count += 1

    for t in text:
        word = t.text
        if word.isalpha():
            word_count += 1
            syllable_count += count_syl(word, d)

    if sentence_count == 0 or word_count == 0:
        return 0.0

    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count

    flesch_score = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59

    return flesch_score


def count_syl(word, d):
    word = word.lower()
    if word in d:
        pronunciation = d[word][0]
        count = 0
        for p in pronunciation:
            if p[-1].isdigit():
                count += 1
        return count
    else:
        vowels = re.findall(r"[aeiouy]+", word)
        return len(vowels)
    pass


def read_novels(path=Path.cwd() / "p1-texts" / "novels"):
    novels = []
    for file in path.glob("*.txt"):
        name_parts = file.stem.split("-")
        if len(name_parts) < 3:
            continue
        
        
        author = name_parts[-2].strip()
        year_str = name_parts[-1].strip()
        title = "-".join(name_parts[:-2]).strip()

        if not year_str.isdigit():
            continue
        
        year = int(year_str)
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            
        book = {
            "text": text,
            "title": title,
            "author": author,
            "year": year
        }
        novels.append(book)
    df = pd.DataFrame(novels)
    df = df.sort_values("year")
    df = df.reset_index(drop=True)
    return df




def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    if not os.path.exists(store_path):
        os.makedirs(store_path)
        
    parsed_t = []
    for t in df["text"]:
        doc = nlp(t)
        parsed_t.append(doc)
        
    df["parsed"] = parsed_t
    save_path = store_path / out_name
    
    f = open(save_path, "wb")
    pickle.dump(df, f)
    f.close()
    return df
    
    pass


def nltk_ttr(text):
    tokens = nltk.word_tokenize(text)
    word_list = []
    for w in tokens:
        if w.isalpha():
            word_list.append(w.lower())
            
    word_count = len(word_list)
    if word_count == 0:
        return 0.0
    
    unique_count = len(set(word_list))
    ttr_value = unique_count / word_count
    return ttr_value


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        fk = fk_level(row["parsed"], cmudict)
        results[row["title"]] = round(fk, 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    total_tokens = len(doc)
    subj_list = []
    all_words = []
    
    for token in doc:
        if token.is_alpha:
            all_words.append(token.text.lower())
            
        if token.lemma_ == target_verb and token.pos_ == "VERB":
            for k in token.children:
                if k.dep_ == "nsubj" or k.dep_ == "nsubjpass":
                    subj_list.append(k.text.lower())
                    
    subj_counts = Counter(subj_list)
    word_counts = Counter(all_words)
    pmi_scores = {}
    
    for s in subj_counts:
        p_subject = word_counts[s] / total_tokens
        p_verb = word_counts[target_verb] / total_tokens
        p_joint = subj_counts[s] / total_tokens
        
        if p_subject > 0 and p_verb > 0 and p_joint > 0:
            pmi = math.log2(p_joint / (p_subject * p_verb))
            pmi_scores[s] = pmi
            
    sorted_pmi = sorted(pmi_scores.items(), key = lambda x: x[1], reverse = True)
    return sorted_pmi[:10]
    



def subjects_by_verb_count(doc, verb):
    subj_list = []
    for token in doc:
        if token.lemma_ == verb and token.pos_ == "VERB":
            kids = token.children
            for k in kids:
                if k.dep_ == "nsubj" or k.dep_ == "nsubjpass":
                    subj_list.append(k.text.lower())
                    
    c = Counter(subj_list)
    return c.most_common(10)
    



def adjective_counts(doc):
    adj_list = []
    for t in doc:
        if t.pos_ == "ADJ":
            adj_list.append(t.text.lower())
            
    counts = Counter(adj_list)
    return counts.most_common(10)




if __name__ == "__main__":
    
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) 
    print(df.head())
    nltk.download("cmudict")
    
    df = parse(df)
    df.to_pickle(Path.cwd() / "pickles" / "parsed.pickle")
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    
    for i, row in df.iterrows():
        print(row["title"])
        print(adjective_counts(row["parsed"]))

    
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    

