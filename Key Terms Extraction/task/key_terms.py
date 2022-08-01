import string
from os import path

from lxml import etree
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    xml_path = path.join(path.dirname(__file__), "news.xml")

    root = etree.parse(xml_path).getroot()
    lemmatizer = WordNetLemmatizer()
    titles_and_words = {}

    # tokenize and lemmatize words
    title_and_token_articles = {item[0].text: word_tokenize(item[1].text.lower()) for item in root[0]}
    for title, token_article in title_and_token_articles.items():
        for token in token_article:
            titles_and_words.setdefault(title, []).append(lemmatizer.lemmatize(token))

    # remove punctuation and stopwords
    for banned_char in [*list(string.punctuation), *stopwords.words('english'), 'last']:
        for title, lemmatized_tokens in titles_and_words.items():
            for token in lemmatized_tokens:
                if banned_char == token:
                    titles_and_words[title].remove(token)

    # remove non-nouns
    for title, words in titles_and_words.items():
        for word in words:
            if pos_tag([word])[0][1] != 'NN':
                titles_and_words[title].remove(word)

    # count the TF-IDF metric for each word in all stories
    dataset = []
    for list_of_words in titles_and_words.values():
        dataset.append(' '.join([word for word in list_of_words]))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset)

    # sort words by their tfidf score (and alphabet) in all texts
    texts_words_scores = {}
    for num, matrix in enumerate(tfidf_matrix.toarray()):
        texts_words_scores[num] = sorted(list(zip(vectorizer.get_feature_names(), tfidf_matrix.toarray()[num])),
                                         key=lambda item: (item[1], item[0]), reverse=True)[:5]

    # print titles and words
    for text, words in texts_words_scores.items():
        print([title for title in titles_and_words.keys()][text] + ':')
        print(*[word for word, tfidf_score in words], '\n')


if __name__ == '__main__':
    main()
