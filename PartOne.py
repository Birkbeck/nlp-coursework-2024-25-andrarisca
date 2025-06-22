#Re-assessment template 2025
# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import re
import pandas as pd 
import pickle
import os


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    raw_lines = nltk.sent_tokenize(text)
    tokens = nltk.word_tokenize(text)
    lines = len(raw_lines)
    vocab_length = 0
    syll_length = 0
    for t in tokens:
        if t.isalpha():
            vocab_length += 1
            syll_length += count_syl(t, d)
            
    if lines == 0 or vocab_length == 0:
        return 0.0
    
    words_line = vocab_length / lines
    syll_word = syll_length / vocab_length
    flesch_level = 0.39 * words_line + 11.8 * syll_word - 15.59
    
    return flesch_level
    pass


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
        
        title = name_parts[0].strip()
        author = name_parts[1].strip()
        year_str = name_parts[2].strip()
        
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
    pass


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
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    pass


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
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

