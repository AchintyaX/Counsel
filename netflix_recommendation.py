import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def recommendation_process(file_path, save_path):
    """
    Processing the file-path
    Doing data cleaning
    Doing feature selection
    Getting Tokenizer ready

    """
    shows_df = pd.read_csv(file_path)

    shows_df.drop(shows_df.columns[[0, 1, 5, 6, 7, 9]], axis=1, inplace=True)

    shows_df.fillna("", inplace=True)

    shows_df[["director", "cast"]] = shows_df[["director", "cast"]].applymap(
        lambda x: " ".join(x.replace(" ", "").split(",")[:3])
    )

    shows_df["title_dup"] = shows_df["title"]

    titles_corpus = shows_df.apply(" ".join, axis=1)

    tfidf_vectorizer_params = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 3), max_df=0.5
    )

    tfidf_vectorizer = tfidf_vectorizer_params.fit_transform(titles_corpus)

    pickle.dump(tfidf_vectorizer, open(save_path, "wb"))

    return f"processed {file_path} and saved tokenizer at {save_path}"


def recommended_shows(title, shows_df, tfidf_vect):

    """
    Recommends the top 5 similar shows to provided show title.
            Arguments:
                    title (str): Show title extracted from JSON API request
                    shows_df (pandas.DataFrame): Dataframe of Netflix shows dataset
                    tfidf_vect (scipy.sparse.matrix): sklearn TF-IDF vectorizer sparse matrix
            Returns:
                    response (dict): Recommended shows and similarity confidence in JSON format
    """

    try:

        title_iloc = shows_df.index[shows_df["title"] == title][0]

    except:

        return "Movie/TV Show title not found. Please make sure it is one of the titles in this dataset: https://www.kaggle.com/shivamb/netflix-shows"

    show_cos_sim = cosine_similarity(tfidf_vect[title_iloc], tfidf_vect).flatten()

    sim_titles_vects = sorted(
        list(enumerate(show_cos_sim)), key=lambda x: x[1], reverse=True
    )[1:6]

    response = {
        "result": [
            {"title": shows_df.iloc[t_vect[0]][0], "confidence": round(t_vect[1], 1)}
            for t_vect in sim_titles_vects
        ]
    }

    return response
