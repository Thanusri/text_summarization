from flask import Flask, request, render_template
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from operator import itemgetter
import numpy as np
import spacy
spacy.load("en_core_web_sm")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/textsummarization',methods=['POST', 'GET'])

def textsummarization():
    #sen = "Computer engineering is a relatively new field of engineering and is one of the fastest growing fields today. Computer engineering is one of today’s most technologically based jobs. The field of computer engineering combines the knowledge of electrical engineering and computer science to create advanced computer systems. Computer engineering involves the process of designing and manufacturing computer central processors, memory systems, central processing units, and of peripheral devices. Computer engineers work with CAD(computer aided design) programs and different computer languages so they can create and program computer systems. Computer engineers use today’s best technology to create tomorrow’s. Computer engineers require a high level of training and intelligence to be skilled at their job. A bachelors degree from a college or university with a good computer engineering program computer science program is necessary. Then once employed their usually is a on the job type of training program to learn the certain types of systems that will be designed and manufactured. Computer engineers major studies conventional electronic engineering, computer science and math in college. The electrical engineering knowledge that a computer engineer possesses allow for a understanding of the apparatus that goes into a computer so that they can be designed and built."
    s = [x for x in request.form.values()]
    sen=''
    for i in s:
        sen+=i
    sentences = list(sen.split('. '))
    stop_words = stopwords.words('english')
    out=''
    for idx, sentence in enumerate(textrank(sentences, stop_words)):
       out=out+str(idx+1)+'. '+''.join(sentence)+'. '
    return render_template('index.html',summary=out)

def textrank(sentences, stopwords=None, top_n=5):
    S = build_similarity_matrix(sentences, stopwords)
    #print("SIMILARITY_MATRIX\n")
    #print(S)
    sentence_ranks = pagerank(S)
    #print("\n\n\nSENTENCE_RANKING\n")
    #print(sentence_ranks)
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:int(top_n)])
    #print("\n\n\nRANKED_INDEXES\n")
    #print(ranked_sentence_indexes)
    summary = itemgetter(*selected_sentences)(sentences)
    #print("\n\n\nRanked_Summary\n")
    return summary
def build_similarity_matrix(sentences, stopwords=None):
    # Create an empty similarity matrix
    S = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue

            S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords)

    # normalize the matrix row-wise
    for idx in range(len(S)):
        S[idx] /= S[idx].sum()
    return S

def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)  # coloumn matrix
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)  # (A^T).p
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P


def sentence_similarity(sent_1, sent_2, stopwords=None):
    # lemmatization
    nlp = spacy.load('en', disable=['parser', 'ner'])
    sent_1 = nlp(sent_1)
    sent_2 = nlp(sent_2)
    sent1 = " ".join([token.lemma_ for token in sent_1])
    sent2 = " ".join([token.lemma_ for token in sent_2])

    # Removig Stop Word
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

if __name__ == "__main__":
    app.run(debug=True)