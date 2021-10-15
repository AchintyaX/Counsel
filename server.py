"""
Usage:
    server.py --file-path=<file-path> --vectorizer=<vectorizer-path>

Options:
    --file-path=<file-path>             file-path of CSV for lookup
    --vectorizer=<vectorizer-path>      file path of vectorizer 

"""

from flask import Flask, app, request, jsonify
from netflix_recommendation import recommended_shows
import pandas as pd
import pickle
from docopt import docopt

app = Flask(__name__)
args = docopt(__doc__)

# Avoid switching the order of 'title' and 'confidence' keys
app.config["JSON_SORT_KEYS"] = False

df = pd.read_csv(args["--file-path"], usecols=[2])
tfidf_vect_pkl = pickle.load(open(args["--vectorizer"], "rb"))

# API endpoint
@app.route("/api/", methods=["POST"])
def process_request():
    # Parse received JSON request
    user_input = request.get_json()

    # Extract show title
    title = user_input["title"]

    # Call recommendation engine
    recommended_shows_dict = recommended_shows(title, df, tfidf_vect_pkl)

    return jsonify(recommended_shows_dict)


if __name__ == "__main__":
    app.run()
