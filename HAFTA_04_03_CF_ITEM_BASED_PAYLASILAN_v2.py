###########################################
# Item-Based Collaborative Filtering (Item-Item Filtering)
###########################################
# Memory-Based Collaborative Filtering: Item Filtering

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: İşlemlerin Fonksiyonlaştırılması

# https://grouplens.org/datasets/movielens/


######################################
# Adım 1: Veri Setinin Hazırlanması
######################################

# Puan verilme alışkanlıkları birbirine benzer olan filmler üzerinden tavsiye sistemi geliştirmek.

import pandas as pd
pd.set_option('display.max_columns', 20)

movie = pd.read_csv('Dersler/HAFTA4/Ders Notları/movie_lens_dataset/movies.csv')
rating = pd.read_csv('Dersler/HAFTA4/Ders Notları/movie_lens_dataset/ratings.csv')
df = movie.merge(rating, how="left", on="movieId")
df.shape

######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

df.shape
df["title"].nunique() #
df["title"].value_counts().head()
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies.shape
rare_movies = comment_counts[comment_counts["title"] <= 1000].index  ##5000
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.head()
common_movies.shape
common_movies["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.head(10)
user_movie_df.columns

len(user_movie_df.columns)
common_movies["title"].nunique()


######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################

movie_name = "Matrix, The (1999)" #korelasyon
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


######################################
# 4. Adım: İşlemlerin Fonksiyonlaştırılması
######################################

#veri okuma bölümümüz
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/ratings.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 5000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df #veri


user_movie_df = create_user_movie_df()

#ilgili film ile diğer filmler arasındaki korelasyon
def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)


movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]


item_based_recommender(movie_name, user_movie_df)

def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Sherlock", user_movie_df)


item_based_recommender("Sherlock Holmes (2009)", user_movie_df)


item_based_recommender(check_film("Sherlock", user_movie_df)[0], user_movie_df)



######################################
# BONUS: USER-MOVIE DF'i Saklama ve Çağırma
######################################


# user_movie_df'in kaydedilmesi
import pickle
pickle.dump(user_movie_df, open("user_movie_df.pkl", 'wb'))

# user_movie_df'inin yüklenmesi
user_movie_df = pickle.load(open('user_movie_df.pkl', 'rb'))

# # tahmin
# movie_name = "Ocean's Twelve (2004)"
# item_based_recommender(movie_name, user_movie_df)