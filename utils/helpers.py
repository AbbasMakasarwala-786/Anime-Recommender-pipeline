import pandas as pd
import numpy as np
import joblib
from config.paths_config import *

################## 1.Get Anime_Frame

def getAnimeFrame(anime,path_df):
    df= pd.read_csv(path_df)
    if isinstance(anime,int):
        return df[df.anime_id==anime]
    if isinstance(anime,str):
        return df[df.eng_version == anime]
    
################ 2.Get_synopsis

def getSynopsis(anime,path_sypnopsis_df):
    sypnopsis_df = pd.read_csv(path_sypnopsis_df)
    if isinstance(anime,int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime]["sypnopsis"].values[0]
    if isinstance(anime,str):
        return sypnopsis_df[sypnopsis_df.Name == anime]["sypnopsis"][0]
    
############### 3. Get Content recommendation 

def find_similar_animes(name,path_anime_weights,path_anime2anime_encoded,
                        path_anime2anime_decoded,path_anime_df,n=10,return_dist=False,negative=False):# negative=True disimilar items ,False Similar animes
    # try:

        anime_weights = joblib.load(path_anime_weights)
        anime2anime_encoded = joblib.load(path_anime2anime_encoded)
        anime2anime_decoded = joblib.load(path_anime2anime_decoded)
        df = pd.read_csv(path_anime_df)


        index = getAnimeFrame(name,path_anime_df).anime_id.values[0]
        
        encoded_index =anime2anime_encoded.get(index)

        weights = anime_weights
        dists = np.dot(weights,weights[encoded_index])
        sorted_dist = np.argsort(dists) # argsort gives the indexes needed for sorting
        n=n+1

        if negative:
            closest = sorted_dist[:n]
        else:
            closest = sorted_dist[-n:]
        
        print(f"Anime closest to {name}")

        if return_dist:
            return dists,closest

        similarityArr =[]
        for closes in closest:
            decoded_id  = anime2anime_decoded.get(closes)

            anime_frame =getAnimeFrame(decoded_id,path_anime_df)
            anime_name =anime_frame.eng_version.values[0]
            genre =anime_frame.Genres.values[0]
            similarity =dists[closes]
            similarityArr.append({
                "anime_id":decoded_id,
                "name":anime_name,
                "similarity":similarity,
                "genre":genre,
                

            })
        Frame =pd.DataFrame(similarityArr).sort_values(by="similarity",ascending=False)
        return Frame[Frame.anime_id!=index].drop(["anime_id"],axis=1)
    # except Exception as e:
    #     print("Error occured",e)


########## 4.Find Similar users
def find_similar_users(item_input,path_user_weights  
                       ,path_user2user_encoded,path_user2user_decoded,n=10,return_dist = False,neg=False):
    try:
        user_weights = joblib.load(path_user_weights)
        user2user_encoded = joblib.load(path_user2user_encoded)
        user2user_decoded = joblib.load(path_user2user_decoded)


        index= item_input
        encoded_index = user2user_encoded.get(index)

        weights =user_weights
        dists = np.dot(weights,weights[encoded_index])
        sorted_dist = np.argsort(dists)

        n=n+1

        if neg:
            closest =sorted_dist[:n]
        else:
            closest= sorted_dist[-n:]
        
        if return_dist:
            return dists,closest
        
        SimilarityArr = []
        for close in closest:
            similarity =dists[close]

            if isinstance(item_input,int):
                decoded_id = user2user_decoded.get(close)
                SimilarityArr.append(
                    {
                        "similar_users":decoded_id,
                        "similarity": similarity
                    }
                )
        similar_users =pd.DataFrame(SimilarityArr).sort_values(by="similarity",ascending=False)
        similar_users = similar_users[similar_users.similar_users != item_input]
        return similar_users

    except Exception as e:
        print("Error",e)


############ 5.Get user prefrences
def get_user_prefrence(user_id,path_rating_df,path_anime_df):

    rating_df=  pd.read_csv(path_rating_df)
    df= pd.read_csv(path_anime_df)

    animes_watched_by_user = rating_df[rating_df.user_id == user_id]
    if animes_watched_by_user.empty:
        print(f"No ratings found for user {user_id}")
        return pd.DataFrame(columns=["eng_version", "Genres"])

    user_rating_percentile =np.percentile(animes_watched_by_user.rating,75)

    animes_watched_by_user = animes_watched_by_user[animes_watched_by_user.rating >=user_rating_percentile]

    top_animes_by_user = (
        animes_watched_by_user.sort_values(by="rating",ascending=False).anime_id.values
    )
    anime_df_rows = df[df["anime_id"].isin(top_animes_by_user)]
    anime_df_rows = anime_df_rows[['eng_version',"Genres"]]
    return anime_df_rows


############ 6.user recommendation

def get_user_recommendation(similar_users,user_pref,path_anime_df,path_sypnopsis_df,path_rating_df,n=10):
    
    df = pd.read_csv(path_anime_df)
    sypnopsis_df = pd.read_csv(path_sypnopsis_df)
    rating_df = pd.read_csv(path_rating_df)


    recommended_animes = []
    anime_list= []

    for user_id in similar_users.similar_users.values:
        pref_list=get_user_prefrence(int(user_id),path_rating_df,path_anime_df)

        pref_list=pref_list[~pref_list.eng_version.isin(user_pref.eng_version.values)]
        if not pref_list.empty:
            anime_list.append(pref_list.eng_version.values)

    if anime_list:
        anime_list = pd.DataFrame(anime_list)

        sorted_list=pd.DataFrame(pd.Series(anime_list.values.ravel()).value_counts()).head(n)

        for i,anime_name in enumerate(sorted_list.index):
            n_user_pref =sorted_list[sorted_list.index == anime_name].values[0][0]

            if isinstance(anime_name,str):
                frame =getAnimeFrame(anime_name,path_anime_df)
                anime_id= frame.anime_id.values[0]
                genre = frame.Genres.values[0]
                synopsis = getSynopsis(int(anime_id),path_sypnopsis_df)
                
                recommended_animes.append(
                    {
                        "n":n_user_pref,
                        "anime_name":anime_name,
                        "Genre":genre,
                        "Synopsis":synopsis
                        
                    }
                )
    return pd.DataFrame(recommended_animes).head(n)


