import json
import time

import redis
import settings
from dependencies import PredictorModels
import logging

#Setting and creating logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#Setting REDIS
db = redis.StrictRedis(
        host=settings.REDIS_IP,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB_ID,
        decode_responses=True
    )


def predict(image_name: str, name_description: str, predictor_categories: PredictorModels):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    name_description : str
        nombre y descripción unidas.

    Returns
    -------
    category_name: dict
        all categories name
    """
    image_path = f'{settings.UPLOAD_FOLDER}/{image_name}'
    logging.info('defining data')
    predictor_categories.define_data(image_path, name_description)
    logging.info('Data was defined successfully')
    logging.info('Preparing data')
    predictor_categories.prepare_data()
    logging.info('Data was prepared succesfully')
    logging.info('Predict categories')
    predictor_categories.predict_categories()
    logging.info('categories was predicted successfully')
    logging.info('Getting results')
    category_name = predictor_categories.get_text_prediction()
    logging.info(f'Result: {category_name}')
    return category_name


def classify_process(predictor_categories: PredictorModels):
    """
    Gets data from REDIS and send it to classify
    
    Adds two integers and returns the result.

    Parameters:
    -----------
    predictor_categories : PredictorModels
        Predictor object.
    """
    while True:
        job_data = db.brpop(settings.REDIS_QUEUE)
        if job_data:
            job_id, data = job_data
            data = json.loads(data)
            job_id = data["id"]
            image_name = data["image_name"]
            name_description = data["text"]
            category_name = predict(image_name, name_description, predictor_categories)
            json_result = json.dumps(category_name)
            db.set(job_id, json_result)
        time.sleep(settings.SERVER_SLEEP)


def create_predictor():
    """
    Creates the predictor object and load all models needed

    Parameters
    ----------
    None

    Returns
    -------
    predictor_categories: PredictorModels
        predictor
    """
    logging.info('Creating predictor object')
    predictor_categories = PredictorModels()
    logging.info('Predictor created successfully')
    logging.info('loading models')
    try:
        predictor_categories.load_models()
        logging.info('models loaded successfully')
    except Exception as e:
        logging.error(f'An error occurred while loading models: {str(e)}')
        raise
    return predictor_categories

if __name__ == "__main__":
    logger.info("Launching ML service...")
    predictor_categories = create_predictor()
    classify_process(predictor_categories)
