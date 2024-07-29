from sklearn.metrics import precision_score, recall_score

# Evaluate recommendation quality
def evaluate_recommendations(user_id, top_n=10):
    recommended_items = recommend_items(user_id, top_n)
    true_items = set(test_data[test_data['userId'] == user_id]['itemId'])
    hits = len(recommended_items & true_items)
    precision = hits / top_n
    recall = hits / len(true_items) if len(true_items) > 0 else 0
    return precision, recall

user_id = ratings['userId'].sample(1).values[0]
precision, recall = evaluate_recommendations(user_id)
print(f'Precision: {precision}, Recall: {recall}')
