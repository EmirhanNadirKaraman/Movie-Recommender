import os

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
import psycopg2

import config
from config import set_values

config.set_values()

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

conn = psycopg2.connect(
    host=os.getenv('URL'),
    database=os.getenv('DATABASE_NAME'),
    user=os.getenv('USERNAME'),
    password=os.getenv('PASSWORD')
)

cursor = conn.cursor()


def add_indices_to_db():
    cursor.execute("ALTER TABLE data ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY")
    conn.commit()

    print("added indices to db")


def get_all_data():
    print("in get all data")
    cursor.execute("SELECT * FROM data")
    result = cursor.fetchall()
    print(result)

    return result


def get_comb_data():
    cursor.execute("SELECT comb FROM data")
    result = cursor.fetchall()
    result = [i[0] for i in result]

    return result


def get_movie_titles():
    cursor.execute("SELECT movie_title FROM data")
    result = cursor.fetchall()
    result = [i[0] for i in result]

    return result


def create_similarity():
    """data = pd.read_csv('main_data.csv')

    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])

    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)

    return data, similarity"""

    data = get_all_data()
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data)
    similarity = cosine_similarity(count_matrix)

    return similarity


def rcmd(movie_title):
    movie_title = movie_title.lower()

    similarity = create_similarity()

    # if movie does not exist in our database then return none
    cursor.execute("SELECT id, movie_title "
                   "FROM data "
                   "WHERE movie_title = %s", (movie_title,))

    if cursor.rowcount == 0:
        return {'success': False, 'message': 'Movie not found in database'}, 400

    result = cursor.fetchone()
    id, title = result[0], result[1]

    lst = list(enumerate(similarity[id]))
    lst = sorted(lst, key=lambda x: x[1], reverse=True)
    lst = lst[1:11]  # excluding first item since it is the requested movie itself

    print('lst', lst)
    # we are trying to find the movie titles of the most similar movies
    result = []

    for i in range(len(lst)):
        a = lst[i][0]
        l.append(data['movie_title'][a])
    return l


# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list


def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())


app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)


@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc) == type('string'):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str


@app.route("/recommend", methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}

    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in
                    range(len(cast_places))}

    reviews_list = []  # list of reviews
    reviews_status = []  # list of comments (good or bad)

    # web scraping to get user reviews from IMDB site
    if imdb_id != "":
        # web scraping to get user reviews from IMDB site
        sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
        soup = bs.BeautifulSoup(sauce, 'html')
        soup_result = soup.find_all("div", {"class": "text show-more__control"})

        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                # passing the review to our model
                movie_review_list = np.array([reviews.string])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                reviews_status.append('Positive' if pred else 'Negative')

    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    # passing all the data to the html file
    return render_template('recommend.html', title=title, poster=poster, overview=overview, vote_average=vote_average,
                           vote_count=vote_count, release_date=release_date, runtime=runtime, status=status,
                           genres=genres,
                           movie_cards=movie_cards, reviews=movie_reviews, casts=casts, cast_details=cast_details)


if __name__ == '__main__':
    add_indices_to_db()
    app.run(debug=True)
