from utils.helpers import *
from config.paths_config import *
from pipeline.prediction_pipeline import hybrid_recommendation
# similar_user = find_similar_users(11800,USER_WEIGHTS_PATH,USER2USER_ENCODED,USER2USER_DECODED)
# user_pref = get_user_prefrence(11880,RATING_DF,DF)
# print(similar_user,user_pref)
print(hybrid_recommendation(11880))