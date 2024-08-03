# Modeling: AI-Powered Recipe Recommender

![model_architecture](https://github.com/user-attachments/assets/2d74c73f-5534-43f5-a126-fbb0c360aa0c)

* **Group 1:** Aktham Almomani, Victor Hsu and Yunus Tezcan
* **Course:** Introduction to Artificial Intelligence (MS-AAI-501) / University Of San Diego
* **Semester:** Summer 2024

## **Introduction**

In this stage, we will be developing an advanced content-based recommendation system for recipes using various machine learning techniques. The goal is to recommend recipes based on the similarity of their ingredients, categories, diet types, and ratings. We will follow these key steps:

* **Data Preprocessing:** Convert the high_level_ingredients list to a single string and combine relevant features into a single text column.
* **TF-IDF Vectorization:** Transform the combined text features into numerical vectors.
* **Dimensionality Reduction:** Use PCA to reduce the dimensionality of the TF-IDF vectors.
* **Clustering:** Apply KMeans clustering to group similar recipes.
* **Neural Network Training:** Train a neural network to predict the cluster of a recipe based on its features.
* **Similarity Calculation:** Implement a function to find and return recipes similar to a given target recipe, weighting the similarity scores by ratings.

## **Dataset**

**[All recipes website](https://www.allrecipes.com/)** is a popular online platform known for its extensive collection of user-generated recipes. It is a go-to resource for home cooks and culinary enthusiasts, offering a diverse range of recipes across various cuisines and dietary preferences. The website features detailed recipe information, including ingredients, instructions, user ratings, and reviews, making it a comprehensive resource for anyone looking to explore new dishes or improve their cooking skills.

For AI-Powered Recipe Recommender project, we will be using a dataset scraped from **[All recipes website](https://www.allrecipes.com/)**. This [dataset](https://github.com/shaansubbaiah/allrecipes-scraper/blob/main/export/scraped-07-05-21.csv), provides a wealth of information about a wide variety of recipes, which will be essential for building an effective recommendation system.

**Key Features of the Dataset:**
* Recipe Titles
* Ingredients
* Instructions
* Ratings
* Reviews
* Preparation and Cooking Times
* Nutritional information

The dataset from **[All recipes website](https://www.allrecipes.com/)** is a rich resource for AI-Recipe Recommender Project. It contains comprehensive details about each recipe, including titles, ingredients, instructions, ratings, reviews, and nutritional information. Leveraging this data, this project can deliver personalized, relevant, and appealing recipe recommendations to users, enhancing their cooking experience and meeting their dietary preferences.

## **Modeling - Advanced Content-Based Recommendation System**

In this modeling phase, we develop a recommendation system to suggest recipes based on user preferences and recipe similarities. The process involves several key techniques and steps to ensure accurate and relevant recommendations:

* **Data Preparation:**
  * High-Level Ingredients: Convert the high_level_ingredients lists into strings for easier processing.
  * Feature Combination: Combine category, diet type, and high-level ingredients into a single text column.
   ```python
      # Convert high_level_ingredients from list to a single string:
       df['high_level_ingredients_str'] = df['high_level_ingredients'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else ' '.join(x))
      # Combine relevant features into a single text column:
      df['combined_features'] = df.apply(lambda row: f"{row['category']} {row['diet_type']} {row['high_level_ingredients_str']}", axis=1)
   ```
* **TF-IDF Vectorization:**
  * Purpose: Transform the combined text features into numerical vectors that represent the importance of each term in the dataset.
  * Technique: Use the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to create a matrix of features.
   ```python
      tfidf = TfidfVectorizer(stop_words='english')
      tfidf_matrix = tfidf.fit_transform(df['combined_features'])
   ```
* **Dimensionality Reduction:**
  * Purpose: Reduce the complexity of the TF-IDF matrix while preserving essential information.
  * Technique: Apply Principal Component Analysis (PCA) to reduce the TF-IDF matrix to 50 components.
   ```python
      pca = PCA(n_components=50)
      tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())
   ```
* **Clustering:**
  * Purpose: Group similar recipes together to facilitate efficient and relevant recommendations.
  * Technique: Use KMeans clustering to divide the dataset into 25 clusters based on the reduced feature set.
   ```python
      kmeans = KMeans(n_clusters=25, random_state=42)
      df['cluster'] = kmeans.fit_predict(tfidf_pca)
   ```
* **Neural Network for Similarity Learning:**
  * Purpose: Learn to predict the cluster membership of recipes, which helps in finding similar recipes.
  * Architecture:
    * Input Layer: 50-dimensional input corresponding to the PCA components.
    * Hidden Layers: Two dense layers with 128 and 64 neurons, respectively, using ReLU activation.
    * Output Layer: 25 output neurons with softmax activation to predict cluster probabilities.
  * Compilation: The model is compiled using the Adam optimizer and categorical cross-entropy loss.
   ```python
      # Neural Network for learning similarity with 25 output neurons:
      input_features = Input(shape=(50,))
      dense1 = Dense(128, activation='relu')(input_features)
      dense2 = Dense(64, activation='relu')(dense1)
      output = Dense(25, activation='softmax')(dense2)
      model = Model(inputs=input_features, outputs=output)
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   ```
* **Model Training:**
  * Data: The PCA-transformed features are used as input (X), and the cluster labels are one-hot encoded as targets (y).
  * Training: The model is trained over 25 epochs with a batch size of 32 and a validation split of 20%.
   ```python
      # Prepare the data for training:
      X = tfidf_pca
      y = tf.keras.utils.to_categorical(df['cluster'], num_classes=25)

      # Train the model:
      model.fit(X, y, epochs=25, batch_size=32, validation_split=0.2)
   ```
* **Recipe Recommendation Function:**
  * Purpose: Generate similar recipe recommendations based on a given recipe.
  * Process:
    * Find Target Cluster: Predict the cluster of the given recipe using the neural network.
    * Calculate Similarities: Use cosine similarity to find recipes within the same cluster.
    * Diversification (Optional): Adjust similarity scores to promote diversity in recommendations.
    * Top Recommendations: Select the top N similar recipes, optionally diversified.
   ```python
      def get_similar_recipes(recipe_name, top_n=5, diversify=False, diversity_factor=0.1):
          target_index = df[df['name'] == recipe_name].index[0]
          target_features = tfidf_pca[target_index].reshape(1, -1)
          target_cluster = model.predict(target_features).argmax()
          cluster_indices = df[df['cluster'] == target_cluster].index
          similarities = cosine_similarity(target_features, tfidf_pca[cluster_indices]).flatten()
          weighted_similarities = similarities * df.loc[cluster_indices, 'rating']
    
          if diversify:
              # Diversify recommendations by adjusting the similarity scores
              diversified_scores = weighted_similarities * (1 - diversity_factor * np.arange(len(weighted_similarities)))
              similar_indices = cluster_indices[np.argsort(diversified_scores)[-top_n:][::-1]]
          else:
              similar_indices = cluster_indices[np.argsort(weighted_similarities)[-top_n:][::-1]]
    
          return df.iloc[similar_indices]
   ```
This comprehensive approach ensures that the recommendation system is both accurate and efficient, providing users with relevant and varied recipe suggestions based on their preferences and the characteristics of the recipes in the dataset.

## **Evaluation of the Recommendation System**

To ensure our recipe recommendation system is effective and meets user needs, we evaluate its performance using three key metrics: Precision, Recall, and Normalized Discounted Cumulative Gain (NDCG). These metrics help us measure the relevance, coverage, and ranking quality of the recommendations.

**Metrics Used for Evaluation**
* Precision:
  * Definition: The proportion of recommended recipes that are relevant.
  * Importance: Ensures users see mostly relevant recipes.
* Recall:
  * Definition: The proportion of all relevant recipes that are recommended.
  * Importance: Ensures users do not miss out on relevant recipes.
* NDCG (Normalized Discounted Cumulative Gain): 
  * Definition: Evaluates the ranking quality of the recommendations, emphasizing top-ranked relevant items.
  * Importance: Ensures the most relevant recipes are presented first.

**Techniques Used for Evaluation**
* Cosine Similarity: Measures the similarity between recipes to identify the most similar ones to a given target recipe.
* Diversification: Ensures recommendations are diverse by adjusting similarity scores to reduce the influence of very similar items.
* Clustering: Groups similar recipes into clusters to facilitate finding and recommending similar recipes.

**Evaluation Process**
* Data Preparation: Split the dataset into training and testing sets.
* Model Training: Train a neural network model to learn recipe similarities based on feature vectors.
* Generating Recommendations: Predict clusters for test recipes, calculate cosine similarities within clusters, and apply diversification if enabled.
Calculating Metrics:

Compute Precision, Recall, and NDCG for the recommendations and average them across all test instances.
The results are compiled into a DataFrame for easy visualization, summarizing the system's performance in relevance, coverage, and ranking quality.

```python
# Evaluation function
def diversify_recommendations(similarities, top_indices, diversity_factor=0.1):
    # Reduce the influence of very similar items
    diversified_scores = similarities * (1 - diversity_factor * np.arange(len(similarities)))
    return np.argsort(diversified_scores)[-len(top_indices):][::-1]

def evaluate_recommendation_system(model, X_test, df, y_test_indices, top_n=5, diversity_factor=0.1):
    # Predict clusters for the test set
    predicted_clusters = model.predict(X_test)
    
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    
    for i, test_idx in enumerate(y_test_indices):
        target_features = X_test[i].reshape(1, -1)
        target_cluster = predicted_clusters[i].argmax()
        cluster_indices = df[df['cluster'] == target_cluster].index
        
        # Filter X_test to only include samples within cluster_indices
        cluster_test_indices = [idx for idx in cluster_indices if idx in y_test_indices]
        
        if not cluster_test_indices:
            continue
        
        test_indices_mapped = [np.where(y_test_indices == idx)[0][0] for idx in cluster_test_indices]
        similarities = cosine_similarity(target_features, X_test[test_indices_mapped]).flatten()
        top_indices = diversify_recommendations(similarities, test_indices_mapped, diversity_factor)
        
        # True labels for evaluation
        true_labels = (y_test[test_indices_mapped] == y_test[i]).astype(int)
        predicted_labels = np.zeros_like(true_labels)
        predicted_labels[top_indices] = 1
        
        # Precision and Recall
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        
        # NDCG
        if np.any(true_labels):
            true_labels_ndcg = np.zeros(len(similarities))
            true_labels_ndcg[top_indices] = true_labels[top_indices]
            similarities_ndcg = np.zeros(len(similarities))
            similarities_ndcg[top_indices] = similarities[top_indices]
            ndcg = ndcg_score([true_labels_ndcg], [similarities_ndcg])
            ndcg_scores.append(ndcg)
    
    avg_precision = np.mean(precision_scores) * 100
    avg_recall = np.mean(recall_scores) * 100
    avg_ndcg = np.mean(ndcg_scores) * 100 if ndcg_scores else 0
    
    return round(avg_precision, 2), round(avg_recall, 2), round(avg_ndcg, 2)

# Split the data
X_train, X_test, y_train_indices, y_test_indices = train_test_split(tfidf_pca, df.index, test_size=0.2, random_state=42)
y_train = df.loc[y_train_indices, 'cluster'].values
y_test = df.loc[y_test_indices, 'cluster'].values

# Train the model on the training set
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=25)
model.fit(X_train, y_train_cat, epochs=25, batch_size=32, validation_split=0.2)

# Evaluate the model
avg_precision, avg_recall, avg_ndcg = evaluate_recommendation_system(model, X_test, df, y_test_indices)

# Create DataFrame to display results
results_df = pd.DataFrame({
    'Metric': ['Average Precision', 'Average Recall', 'Average NDCG'],
    'Score (%)': [avg_precision, avg_recall, avg_ndcg]
})
results_df

```


## **Evaulation Summary Highlights:**

| Metric             | Score (%) |
|--------------------|-----------|
| **Average Precision** | 96.92     |
| **Average Recall**    | 96.92     |
| **Average NDCG**      | 100.00    |

* **Average Precision:** Of all the recipes recommended by the system, `96.92%` were relevant. This indicates a high level of accuracy in the recommendations provided, meaning users are likely to find the suggested recipes relevant to their interests.
* **Average Recall:** The system was able to identify `96.92%` of all relevant recipes in the dataset. This high recall suggests that the recommendation system is comprehensive in retrieving almost all relevant items, ensuring users do not miss out on potentially interesting recipes.
* **Average NDCG:** The Normalized Discounted Cumulative Gain (NDCG) score of `100.00%` indicates perfect ranking quality. This means that the most relevant recipes are consistently placed at the top of the recommendation list, enhancing user satisfaction by ensuring the best options are always presented first.

**Overall Performance**
* High Precision and Recall: The system demonstrates both high precision and high recall, indicating that it effectively identifies and recommends relevant recipes.
* Perfect Ranking: The perfect NDCG score reflects the system's ability to rank the recommendations accurately, presenting the most relevant recipes first.

These metrics suggest that the recommendation system is performing exceptionally well. Users are likely to experience highly relevant and well-ranked recipe recommendations, leading to a positive user experience.
