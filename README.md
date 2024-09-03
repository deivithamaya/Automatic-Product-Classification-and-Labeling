# Final-Project Anyone-AI 
This API will help you to classify products using AI

# Python version
API:
`3.12`
Model:
`3.8`

## Requirements for development and execution

* Python
* pip
* Docker
* Docker Compose

## Environment variables
It is necessary to save an .env file where variables needed exists


## To set up the project on site
> `docker`and `docker-compose`must be installed

To build this project it is necessary to set all the requirements.

An .env with the necessary variables, the categories.json in the next path:

model/categories

The fusion or text models for all 5 categories using the next path:

    * model/models_conf/fusion_models -> For fusion models
    * model/models_conf/text_models -> For text models

And finally, the following command:

```
docker-compose up --build -d
```

## Services used with Docker

-redis
-ML deploy
-Flask app


## Other scripts

There are two folders called "EDA" and "model_training". Those folders have the data analysis and the model creation and training. Every folder has an README to run. And, it is necessary to read everyone to run the project correctly. 

This README has just the information about the services related to the model integrated to an API. 