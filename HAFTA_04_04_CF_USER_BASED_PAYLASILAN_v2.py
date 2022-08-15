############################################
# User-Based Collaborative Filtering
#############################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#sinanın izlediği filmleri belirleyeceğiz

# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#sinanla aynı filmi izleyenler kim idlerine ulaşıcaz

# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#aynı filmi izlemiş olabilir ama sinanla aynı beğenme davranışına sahip olmayabilir
#sinanla aynı filmi izleyenlerin aynı beğenme davranışına sahip olanlarına veriyi indirgiyecegiz

# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#silmlere göre sinana  benzer kullanıcıların verdiği skorları
# ve sinanla olan  corelasyonları çarpıp bir scora dönüştürüp bunu üzerinden sinana
#hangi filmleri önermemiz gerektiğini ortaya çıkarıcaz




# Adım 6: Çalışmanın Fonksiyonlaştırılması


# Kullanıcıların davranış benzerlikleri üzerinden film önerileri yapılacak.


#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################

movie = pd.read_csv('Dersler/HAFTA4/Ders Notları/movie_lens_dataset/movies.csv')
rating = pd.read_csv('Dersler/HAFTA4/Ders Notları/movie_lens_dataset/ratings.csv')
df = movie.merge(rating, how="left", on="movieId")
df.shape

###5000 beğeniden az olan filmleri çıkardık

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Dersler/HAFTA4/Ders Notları/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Dersler/HAFTA4/Ders Notları/movie_lens_dataset/ratings.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 5000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


# import pickle
# user_movie_df = pickle.load(open('user_movie_df.pkl', 'rb'))

import pandas as pd
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist() #sinanın izleidiği filmler

user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Speed (1994)"]

len(movies_watched) #32 tane izlemiş bu kişi


#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

pd.set_option('display.max_columns', 5)


movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

user_movie_count = movies_watched_df.T.notnull().sum()
#kullanıcıların toplam kaç tane film izlediği  bilgisine ulaştık

user_movie_count = user_movie_count.reset_index()


user_movie_count.columns = ["userId", "movie_count"]


user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

user_movie_count[user_movie_count["movie_count"] == 32].count()

# perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

users_same_movies.count()

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]



#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################
#aynı filmi izlemiş olabilir ama sinanla aynı beğenme davranışına sahip olmayabilir
#sinanla aynı filmi izleyenlerin aynı beğenme davranışına sahip olanlarına veriyi indirgiyecegiz



# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız


final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

final_df.shape


final_df.T.corr()


corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

corr_df["corr"].describe()

# Sinanla yüzde 65 ve üzeri korelasyona sahip kullanıcılar:
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.50)][
    ["user_id_2", "corr"]].reset_index(drop=True)


top_users = top_users.sort_values(by='corr', ascending=False)


top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv('Dersler/HAFTA4/Ders Notları/movie_lens_dataset/ratings.csv')

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings=top_users_ratings[top_users_ratings["userId"]!=random_user] #sinanı çıkardık


#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################

# weighted_rating'in hesaplanması.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df["weighted_rating"].describe()
recommendation_df[recommendation_df["weighted_rating"] > 3]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('Dersler/HAFTA4/Ders Notları/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])
#sinanla en benzer davranış sergileyen bir çok kulannıcının cor ve ratingleeri göz önünde bulundurularak ağırlık skorlarına göre önerilmesi beklenen filmlerdir.

#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

def user_based_recommender():
    import pickle
    import pandas as pd
    user_movie_df = pickle.load(open('user_movie_df.pkl', 'rb'))
    random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 4].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])


user_based_recommender()


#############################################
# PROJE: Hybrid Recommender System
#############################################



























