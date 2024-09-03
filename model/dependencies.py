# Standard Python libraries
import os
import json

# Third-party libraries
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import logging

# Project-specific libraries
import settings
import utils

#Setting and creating logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictorModels:
    """
    Fusion of text and image models for making predictions
    """


    def __init__(self) -> None:
        self.is_fusion = [False,False,False,False,False]
        self.predictions = []
        self.models = []


    def define_data(self, image_path: str, text_name_description: str):
        """
        Set data to get prediction

        Parameters
        --------------
        image_path: str
            The image path where image is saved

        text_name_description: str
            The name and description about product
        """
        self.image_path = image_path
        self.text_name_description = text_name_description


    def _get_models_found(self, level):
        main_root = 'models_conf/fusion_models'
        list_dir = os.listdir(main_root) 
        return (f'configL{level}.json' in list_dir) and (f'weightsL{level}.h5' in list_dir)

        
    def _get_model_keras(self, level, is_fusion_level):
        if is_fusion_level: root = 'models_conf/fusion_models'
        else: root = 'models_conf/text_models'
        config_root = root + f'/configL{level}.json'
        weights_root = root + f'/weightsL{level}.h5'
        with open(config_root, 'r') as file_json:
            model =  tf.keras.models.model_from_json(json.dumps(json.load(file_json)))
        model.load_weights(weights_root)
        return model


    def _preprocess_image(self):
        """
        Preprocess image to create input data
        """
        raw = tf.io.read_file(self.image_path)
        tensor = tf.image.decode_jpeg(raw, channels=3)
        tensor2 = tf.image.resize(tensor, [224, 224], preserve_aspect_ratio=True)
        shape = tf.shape(tensor2)
        h = (224 - shape[0]) //2
        w = (224 - shape[1]) //2
        h = tf.maximum(h, 0)
        w = tf.maximum(w, 0)
        tensor = tf.image.pad_to_bounding_box(tensor2, int(h), int(w), 224, 224)
        mask = tf.image.pad_to_bounding_box(tf.ones_like(tensor2), h, w, 224, 224)
        tensor = tf.cast(tensor, tf.float32) / 255.0
        tensor = tensor * mask + (1 - mask) * 0.5
        self.image_processed = np.array([tensor])


    def _preprocess_text(self):
        """
        Preprocess text to create input data
        """
        toke = self.preprocessor([self.text_name_description])
        self.text_processed = self.bert_model(toke)['pooled_output']
    

    def load_models(self):
        self.bert_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3", trainable=False)
        self.preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        for level in range(1,6):
            logging.info(f'level: {level}')
            is_fusion_level = self._get_models_found(level)
            self.is_fusion[level-1] = is_fusion_level
            model = self._get_model_keras(level, is_fusion_level)
            self.models.append(model)

    
    def prepare_data(self):
        self._preprocess_image()
        self._preprocess_text()


    def predict_categories(self):
        for index, element in enumerate(self.is_fusion):
            if element: input_data = [self.text_processed, self.image_processed]
            else: input_data = self.text_processed
            model = self.models[index]
            predict = model.predict(input_data)
            self.predictions.append(predict[0])

        
    def get_text_prediction(self):
        category_level_1,category_level_2,category_level_3,category_level_4,category_level_5 = utils.process_predictions(
            self.predictions[0],
            self.predictions[1],
            self.predictions[2],
            self.predictions[3],
            self.predictions[4]
        )
        category_name = {
            "one": category_level_1,
            "two": category_level_2,
            "three": category_level_3,
            "four": category_level_4,
            "five": category_level_5
        }
        return category_name
