import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation,BatchNormalization,Input,Embedding,Dot,Dense,Flatten
from utils.common_functions import read_yaml
from src.logger import get_logger
from src.custom_exception import CustomException

logger=get_logger(__name__)

class BaseModel:
    def __init__(self,config_path):
        try:
            self.config =read_yaml(config_path)
            logger.info("Loaded coniguration from config.yaml")
        except Exception as e:
            raise CustomException("Error loading the Config file from configurations")
        
    def RecommenderNet(self,n_users,n_anime):
        try:
            embedding_size = self.config['model']['embedding_size']

            user =Input(name="user",shape=[1])
            user_embedding =Embedding(name="user_embeddings",input_dim=n_users,output_dim=embedding_size)(user)

            anime=Input(name="anime",shape=[1])
            anime_embedding =Embedding(name="anime_embedding",input_dim=n_anime,output_dim=embedding_size)(anime)

            x=Dot(name="Dot_Product",normalize=True,axes =2)([user_embedding,anime_embedding])# dot product to find similarity between user and anime

            x= Flatten()(x)
            x= Dense(1,kernel_initializer='he_normal')(x)
            x=BatchNormalization()(x)
            x=Activation("sigmoid")(x)

            model =Model(inputs=[user,anime],outputs=x)
            model.compile(loss=self.config['model']['loss'],
                            metrics=self.config['model']['metrics'],
                            optimizer=self.config['model']['optimizer'])
            logger.info("Model created successfully")
            return model

        except Exception as e:
            logger.error(f"Error occured during model building {e}")
            raise CustomException("Failed to create model")




