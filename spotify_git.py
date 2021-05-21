import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from openpyxl.workbook import Workbook
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st
import streamlit.components.v1 as componnents

df=pd.read_csv('spotify_dataset_f.csv',header=0)
X=df.drop(['popularity'],axis=1)
y=df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LinearRegression()
clf.fit(X_train, y_train)
pickle.dump(clf,open('spotify.pkl','wb'))
loaded_model = pickle.load(open('spotify.pkl', 'rb'))

#####################################Recommendations####################################################
dataf=pd.read_csv('recommendations.csv',header=0)
print(dataf.columns)
x=dataf[dataf.drop(columns=['artists','name']).columns].values
scaler =StandardScaler().fit(x)
X_scaled = scaler.transform(x)
dataf[dataf.drop(columns=['artists','name']).columns]=X_scaled
dataf.sort_values('tempo',inplace=True)


def find_song(song_name, df, number=10):
    song_names = df['name'].values
    artists = df['artists'].values
    song_list = []
    count = 0

    #if song_name[-1] == ' ':
    #    song_name = song_name[:-1]
    for i in song_names:
        if song_name.lower() in i.lower():
            song_list.append([len(song_name) / len(i), count])
        else:
            song_list.append([0, count])
        count += 1

    song_list.sort(reverse=True)  # list containing list of len(song_name)/len(i) value and count (row number)
    s = [[song_names[song_list[i][1]], artists[song_list[i][1]].strip('][').split(', ')] for i in
         range(number)]  # list containing list of song name and its artist name/names
    songs = [song_names[song_list[i][1]] for i in range(number)]  # list containing just the song names
    artist = [artists[song_list[i][1]] for i in range(number)]  # list containing just the artist names

    x = []
    for i in s:
        l = ''
        by = ''
        for j in i[1]:
            by += j
        l += i[0] + ' by ' + by
        x.append(l)  # list of strings containing song names and artists in the form "song_name by artist_name"

    slist = []  # this will be a list containing tuples of songs with name equal to what user has entered
    for i in range(number):
        slist.append((x[i], i))  # appending song and its artists with an index as a tuple

    return slist, songs, artist


def find_cos_dist(df, song, number, artist, st):
    x = df[(df['name'] == song) & (df['artists'] == artist)].drop(
        columns=['name', 'artists']).values  # vector for the user entered song

    artist = artist.replace("'", "").replace("'", "").replace('[', '').replace(']', '')
    if ',' in artist:
        inm = artist.rfind(",")
        artist = artist[:inm] + ' and' + artist[inm + 1:]

    song_names = df['name'].values
    p = []
    count = 0

    for i in df.drop(columns=['artists', 'name']).values:
        p.append([distance.cosine(x, i), count])
        count += 1
    p.sort()  # list of all cosine distances with row count
    st.header('The songs closest to your search ' + song + ' by ' + artist + ' :')
    for i in range(1, number + 1):
        artists = dataf['artists'].values
        artist = artists[p[i][1]]
        artist = artist.replace("'", "").replace("'", "").replace('[', '').replace(']', '')
        if ',' in artist:  # dealing with multiple artists
            inm = artist.rfind(",")
            artist = artist[:inm] + ' and' + artist[inm + 1:]
        st.subheader(song_names[p[i][1]] + ' - '+ artist)

def spotify_show():      ## UI for user input uses streamlit
    st.set_page_config(layout="wide")
    st.title("Spotify - Dashboard, Prediction and Song Recommendation")
    componnents.iframe("https://datastudio.google.com/embed/reporting/56646c03-a0f8-41f2-99b7-f253078b0faf/page/AUKyB",
                       height=600, width=1000)

    def get_feedback():
        return []

    st.sidebar.header("Dear user, please give your valuable feedback:")
    name = st.sidebar.text_input("Name")
    dash = st.sidebar.slider("Rate the dashboard", 1, 10)
    pred = st.sidebar.selectbox("Are you satisfied with the prediction results?", ['Yes', 'No'])
    recom = st.sidebar.radio("Are you satisfied with the song recommendations?", ['Yes', 'No'])
    recommend = st.sidebar.selectbox("Would you recommend this webapp to others?", ['Yes', 'No'])
    if st.sidebar.button("Submit"):
        if name != '' and name.isalpha():
            get_feedback().append({"Name": name, "Dashboard Rating": dash, "Prediction satisfaction": pred,
                                   "Recommendation satisfaction": recom, "Recommend others": recommend})
            st.sidebar.success(f'Thanks for visiting this webpage, {name} :)')
        else:
            st.sidebar.error('Please enter a valid name!')
    ddff=pd.DataFrame(get_feedback())
    ddff.to_excel("set_feedback.xlsx")
    ddff.to_csv("set_feedback.csv")

    st.header("Prediction")
    year = st.slider("Year", min_value=1921, max_value=2020)
    st.text("Year in which the song was released")
    danceability = st.slider("Danceability", min_value=0.0, max_value=1.0)
    st.text('One a scale of 0-1, how much suitable the song is to dance')
    energy = st.slider("Energy", min_value=0.0, max_value=1.0)
    st.text('One a scale of 0-1, what is the intensity level of the song')
    artists_popularities = st.slider("Artist Popularity", min_value=0.0, max_value=100.0)
    st.text('One a scale of 0-1, what is the popularity level of the artist')
    loudness = st.number_input("Loudness", min_value=-100.0, max_value=50.0)
    st.text('One a scale of -100 to 50, what is the loudness level of the song')
    tempo = st.number_input("Tempo", min_value=0.0, max_value=250.0)
    st.text('One a scale of 0-250, what is the tempo(in beats per minute) of the song')
    predict = st.button("Predict Popularity")

    if predict:
        test_data = np.array([[year, danceability, energy, loudness, tempo, artists_popularities]])
        output = loaded_model.predict(test_data)[0]
        st.subheader('Popularity score of the song (out of 100):')
        st.success(output)

    st.header("Recommendation")
    song_name = st.text_input("Enter the name of the song you like")
    no_of_recom = st.slider("The number of recommendations you want", 1, 20)
    tup, s, ar = find_song(song_name, dataf)
    if song_name:
        st.subheader(f'Closest songs to-> {song_name}:')
        st.text('You can make the song choice from the below dropdown')
        xx = st.selectbox("", options=tup, key='ishu', )
        if xx:
            find_cos_dist(dataf, s[xx[1]], no_of_recom, ar[xx[1]], st)


spotify_show()














