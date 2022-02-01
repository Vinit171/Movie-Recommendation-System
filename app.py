from flask import Flask, render_template, request
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

l=[]
app = Flask(__name__)

@app.route('/',methods = ["GET","POST"])
def hello_world():
    #global movie_nameee

    if request.method == "POST":
        myDict = request.form
        #title = request.form.get('title')
        title =(myDict['title'])
        age = (myDict['age'])
        gender = (myDict['gender'])


        df = pd.read_csv('p6.csv')

        selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

        for feature in selected_features:
            df[feature] = df[feature].fillna('')

        combined_features = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df[
            'director']

        vectorizer = TfidfVectorizer()

        feature_vectore = vectorizer.fit_transform(combined_features)

        similarity = cosine_similarity(feature_vectore)

        moviee_name = title
        important = [title,age,gender]
        print(important)
        #print(moviee_name)

        list_of_all_titles = df['title'].tolist()
        #print(list_of_all_titles )
        find_close_match = difflib.get_close_matches(moviee_name, list_of_all_titles)
        print(find_close_match)
        close_match = find_close_match[0]
        
        index_of_the_movie = list_of_all_titles.index(close_match)

        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # print the name of similar movies based on the index

        print('Movies suggested for you :\n')
        i = 1

        for moviess in sorted_similar_movies:
            index = moviess[0]
            title_from_index = df[df.index == index]['title'].values[0]
            #print(title_from_index)
            # is line ka mtlb ke df dataset mei , if df.index ke value index se match kr gyi toh title ke
            # values print kr deni hai

            if (i < 16):

                #print(i, "-", title_from_index)
                i = i + 1
                l.append(title_from_index)
                print(title_from_index)

        qqq = ' ,'.join(l)
        #print(qqq)
        return render_template('show.html', inf=qqq)

    return render_template('kd.html')


if __name__=="__main__":
    app.run(debug=True)