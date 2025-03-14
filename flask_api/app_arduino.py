from flask import Flask
from flask_cors import CORS
import logging
from routes_arduino import routes  # Import after initializing Flask

# Enable full error logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for frontend communication
CORS(app)

# Register routes from routes.py
app.register_blueprint(routes)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)