import nltk
import sys
import math
from collections import Counter
import glob
import string


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Returns mapping for each txt file
    corpus = {}

    for file_path in glob.glob(f"{directory}/*.txt"):
        with open(file_path, "r", encoding='utf8') as file:
            file_name = file_path.split("/")[-1]
            corpus[file_name] = file.read()

    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punctuation = string.punctuation
    stop_words = nltk.corpus.stopwords.words("english")

    # Process document by coverting all words to lowercase
    words = nltk.word_tokenize(document.lower())
    # Removes any punctuation or English stopwords
    words = [word for word in words if word not in punctuation and word not in stop_words]

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Given a dictionary of `documents` that maps names of documents to a list
    # of words, return a dictionary that maps words to their IDF values.

    N = len(documents)
    word_counts = Counter(word for doc in documents.values() for word in doc)
    idf = {}
    # Gets at least one of the documents from dictionary

    for word, count in word_counts.items():
        idf[word] = math.log(N / count)
    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # uses a dictionary comprehension to calculate the scores for each file in one line, rather than a loop
    scores = {filename: sum(idfs.get(word, 0) * filecontent.count(word)
                            for word in query if word in filecontent) for filename, filecontent in files.items()}
    return sorted(scores, key=scores.get, reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = list()

    for sentence in sentences:
        sentence_val = [sentence, 0, 0]

        for word in query:
            if word in sentences[sentence]:
                # Compute “matching word measure”
                sentence_val[1] += idfs[word]
                # Compute "query term density"
                sentence_val[2] += sentences[sentence].count(word) / len(sentences[sentence])

        sentence_scores.append(sentence_val)

    return [sentence for sentence, mwm, qtd in sorted(sentence_scores, key=lambda item: (item[1], item[2]), reverse=True)][:n]


if __name__ == "__main__":
    main()
