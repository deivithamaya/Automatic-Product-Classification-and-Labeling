import logging
from logging.handlers import RotatingFileHandler

from flask import Flask

import settings
from views import router



app = Flask(__name__)

# Se rotative files handler
file_handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
file_handler.setLevel(logging.INFO)

# Set console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Log format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add Flask logger's handlers
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)


app.config['UPLOAD_FOLDER'] = settings.UPLOAD_FOLDER
app.secret_key = "secret key"
app.register_blueprint(router)

if __name__ == '__main__':
    app.run(debug=settings.API_DEBUG)
