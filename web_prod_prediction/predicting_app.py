

from flask import Flask, render_template, request
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD

app = Flask(__name__)

# Load   pre-trained model and dataset
 
MODEL_PATH = '/home/liebera6/mysite/collab_filtering_model.pkl'
DATASET_PATH = '/home/liebera6/mysite/merged_data.csv'

# Load the Surprise model
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Load   dataset into a DataFrame (assuming it has 'customer_id' and 'product_id' columns)
data = pd.read_csv(DATASET_PATH)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    # Get a list of unique product IDs
    unique_product_ids = data['product_id'].unique()
    # Make predictions for the user
    user_predictions = []
    for item_id in unique_product_ids:
        pred = model.predict(user_id, item_id)
        user_predictions.append({
            'item_id': item_id,
            'predicted_rating': pred.est
        })

    # Sort predictions by predicted rating in descending order
    user_predictions = sorted(user_predictions, key=lambda x: x['predicted_rating'], reverse=True)

    # Get top recommended products (you can adjust how many you want to display)
    top_recommendations = user_predictions[:5]  # Display top 5 recommendations

    return render_template('recommendations.html', user_id=user_id, recommendations=top_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
