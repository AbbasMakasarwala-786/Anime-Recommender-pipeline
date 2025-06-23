from config.paths_config import *
from utils.helpers import *

def hybrid_recommendation(user_id,user_weight = 0.5,content_weight = 0.5):


    ## Collabarative filtering
    similar_users = find_similar_users(11800,USER_WEIGHTS_PATH,USER2USER_ENCODED,USER2USER_DECODED)
    user_pref = get_user_prefrence(11880,RATING_DF,DF)
    user_recommended_anime_list = get_user_recommendation(similar_users,user_pref,DF,SYNOPSIS_DF,RATING_DF)


    user_recommended_anime_list = user_recommended_anime_list["anime_name"].to_list()
    print(user_recommended_anime_list)
    # content recommendation
    content_recommendation_animes =[]
    for anime in user_recommended_anime_list:
        similar_animes = find_similar_animes(anime,ANIME_WEIGHTS_PATH,ANIME2ANIME_ENCODED,ANIME2ANIME_DECODED,DF,n=10)

        if similar_animes is not None and not similar_animes.empty:
            content_recommendation_animes.extend(similar_animes["name"].tolist())
        
        else:
            print(f"No similar anime found{anime}")
    
    combined_scores ={}

    for anime in user_recommended_anime_list:
        combined_scores[anime] = combined_scores.get(anime,0)+user_weight
    
    for anime in content_recommendation_animes:
        combined_scores[anime]= combined_scores.get(anime,0) + content_weight

    sorted_animes = sorted(combined_scores.items(),key=lambda x:x[1],reverse=True)
    return [anime for anime,score in sorted_animes[:10]]