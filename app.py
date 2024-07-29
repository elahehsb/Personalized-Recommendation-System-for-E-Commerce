from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)

# Load pre-trained models and data
products = pd.read_csv('path/to/products.csv')
user_item_matrix = pd.read_csv('path/to/user_item_matrix.csv', index_col=0)
item_matrix = np.load('path/to/item_matrix.npy')
user_matrix = np.load('path/to/user_matrix.npy')
tfidf_vectorizer = joblib.load('path/to/tfidf_vectorizer.joblib')
cosine_sim = np.load('path/to/cosine_sim.npy')

def recommend_items(user_id, top_n=10):
    user_ratings = user_item_matrix.loc[user_id]
    predictions = user_ratings.index.map(lambda item: predict_rating(user_id, item))
    recommended_items = pd.Series(predictions, index=user_ratings.index).sort_values(ascending=False).head(top_n)
    return recommended_items.index

def recommend_content_based(item_id, top_n=10):
    idx = products.index[products['itemId'] == item_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    item_indices = [i[0] for i in sim_scores]
    return products['itemId'].iloc[item_indices]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data.get('user_id')
    item_id = data.get('item_id')
    
    if user_id:
        recommended_items = recommend_items(user_id)
    elif item_id:
        recommended_items = recommend_content_based(item_id)
    else:
        return jsonify({'error': 'No user_id or item_id provided'}), 400
    
    return jsonify({'recommended_items': recommended_items.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
