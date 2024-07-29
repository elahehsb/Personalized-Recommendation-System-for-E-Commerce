from sklearn.decomposition import NMF

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='itemId', values='normalized_rating').fillna(0)

# Matrix Factorization using NMF
nmf = NMF(n_components=20, random_state=42)
user_matrix = nmf.fit_transform(user_item_matrix)
item_matrix = nmf.components_

# Predict ratings
def predict_rating(user_id, item_id):
    user_idx = user_item_matrix.index.get_loc(user_id)
    item_idx = user_item_matrix.columns.get_loc(item_id)
    return np.dot(user_matrix[user_idx], item_matrix[:, item_idx])

# Generate recommendations
def recommend_items(user_id, top_n=10):
    user_ratings = user_item_matrix.loc[user_id]
    predictions = user_ratings.index.map(lambda item: predict_rating(user_id, item))
    recommended_items = pd.Series(predictions, index=user_ratings.index).sort_values(ascending=False).head(top_n)
    return recommended_items.index
