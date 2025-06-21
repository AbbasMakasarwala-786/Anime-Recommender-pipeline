import os 
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml


logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in given path{file_path}")
        with open(file_path,"r") as yaml_file:
            config =yaml.safe_load(yaml_file)
            logger.info("Succesfully read the YAML file")
            return config
        
    except Exception as e:
        logger.error("Error while reading the yaml file")
        raise CustomException("Failed to read yaml file",e)