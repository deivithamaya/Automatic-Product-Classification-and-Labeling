import json
import time
from uuid import uuid4
from flask import current_app

import redis
import settings

db = redis.StrictRedis(host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID, decode_responses=True)


def model_predict(image_name, name, description):
    """
    Gets data and send it to REDIS

    Parameters
    ----------
    image_name : str
        Image filename.

    name : str
        Product name

    description : str
        Product description ¿

    Returns
    -------
    categories: dict
        all categories name
    """
    categories = None
    job_id = str(uuid4())
    all_name = name.lstrip() + ' ' + description.lstrip()
    job_data = {
        "id": job_id,
        "image_name": image_name,
        "text": all_name
    }
    db.lpush(settings.REDIS_QUEUE, json.dumps(job_data))

    while True:
        output = db.get(job_id)
        if output is not None:
            output = json.loads(output)
            categories = output
            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return categories
