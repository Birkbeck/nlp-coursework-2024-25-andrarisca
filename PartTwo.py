import csv
from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



def rename_labour_co_op(csv_reader):
    updated_rows = []
    for (_, row) in enumerate(csv_reader):
        if row['party'] == "Labour (Co-op)":
            row.update({'party': "Labour"})
            updated_rows.append(row)
        else:
            updated_rows.append(row)
    return updated_rows

def count_parties(csvRows):
    partyCount = dict()
    
    for row in csvRows:
        party_name = row['party']
        if partyCount.get(party_name, None) == None:
            partyCount.update({party_name: 1})
        elif partyCount.get(party_name, None) != None:
            partyCount[party_name] += 1
            
    return partyCount

most_common_party_name = {
    'Conservative': 25079,
    'Labour': 8038,
    'Scottish National Party': 2303,
    'Liberal Democrat': 803,
}

def delete_uncommon_parties(csvRows):
    updated_rows = []
    for row in csvRows:
        if row['party'] in most_common_party_name.keys():
            updated_rows.append(row)
    return updated_rows

def main():
    
    with open('p2-texts/hansard40000.csv', 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file)
        renamed = rename_labour_co_op(reader)
        print("After renaming:", len(renamed))
        cleaned_up = delete_uncommon_parties(renamed)
        print("After cleaning:", len(cleaned_up))
        
        df = pd.DataFrame(cleaned_up)
        print("Shape after converting to DataFrame:", df.shape)

        df = df[df['speech_class'] == 'Speech']
        print("Shape after filtering speech_class:", df.shape)

        df = df[df['speech'].str.len() >= 1000]
        print("Shape after filtering short speeches:", df.shape)

        df = df.reset_index(drop=True)

        df.to_csv("cleaned_speeches.csv", index=False)
        print("Saved cleaned_speeches.csv")


main()

def part_two_b():
    df = pd.read_csv("cleaned_speeches.csv")
    print("Clean file shape:", df.shape)
    
    tfidf = TfidfVectorizer(stop_words = 'english', max_features=3000)
    X = tfidf.fit_transform(df['speech'])
    print("TF-IDF matrix shape:", X.shape)
    y = df['party']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=26
    )
    print("Train set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    return X_train, X_test, y_train, y_test

part_two_b()

def part_two_c():
    df = pd.read_csv("cleaned_speeches.csv")
    print("Clean file shape:", df.shape)
    
    tfidf = TfidfVectorizer(stop_words = 'english', max_features=3000)
    X = tfidf.fit_transform(df['speech'])
    
    y = df['party']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=26
    )
    
    
    rf = RandomForestClassifier(n_estimators = 300)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average = 'macro')
    print("\nRandom Forest Macro F1 Score:", f1_rf)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, zero_division=0))
    
    svm = SVC(kernel = 'linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    f1_svm = f1_score(y_test, y_pred_svm, average = 'macro')
    print("\nSVM Macro F1 Score:", f1_svm)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, zero_division=0))
    
part_two_c()

def part_two_d():
    df = pd.read_csv("cleaned_speeches.csv")
    print("Clean file shape:", df.shape)
    
    tfidf = TfidfVectorizer(
        stop_words = 'english',
        max_features = 3000,
        ngram_range =(1,3)
    )
        
    
    X = tfidf.fit_transform(df['speech'])
    
    y = df['party']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=26
    )
    
    
    rf = RandomForestClassifier(n_estimators = 300)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average = 'macro')
    print("\nRandom Forest Macro F1 Score:", f1_rf)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    svm = SVC(kernel = 'linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    f1_svm = f1_score(y_test, y_pred_svm, average = 'macro')
    print("\nSVM Macro F1 Score:", f1_svm)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, zero_division=0))


    
part_two_d()

stop_words = set(stopwords.words('english'))
def my_tokenizer(text):
    tokens = word_tokenize(text)
    cleaned = []
    for token in tokens:
        if token.isalpha() and token.lower() not in stop_words:
            cleaned.append(token.lower())
    return cleaned

def part_two_e():
    df = pd.read_csv("cleaned_speeches.csv")
    print("Clean file shape:", df.shape)
    
    tfidf = TfidfVectorizer(
        tokenizer = my_tokenizer,
        max_features = 3000
    )
    X = tfidf.fit_transform(df['speech'])
    
    
    y =df['party']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=26
    )
    
    rf = RandomForestClassifier(n_estimators=300)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    print("\nRandom Forest Macro F1 Score with custom tokenizer:", f1_rf)
    print("Random Forest Classification Report:")
    
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro')
    print("\nSVM Macro F1 Score with custom tokenizer:", f1_svm)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, zero_division=0))

    
part_two_e()
