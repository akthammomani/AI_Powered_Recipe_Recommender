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
* **TF-IDF Vectorization:**
  * Purpose: Transform the combined text features into numerical vectors that represent the importance of each term in the dataset.
  * Technique: Use the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to create a matrix of features.
* **Dimensionality Reduction:**
  * Purpose: Reduce the complexity of the TF-IDF matrix while preserving essential information.
  * Technique: Apply Principal Component Analysis (PCA) to reduce the TF-IDF matrix to 50 components.
* **Clustering:**
  * Purpose: Group similar recipes together to facilitate efficient and relevant recommendations.
  * Technique: Use KMeans clustering to divide the dataset into 25 clusters based on the reduced feature set.
* **Neural Network for Similarity Learning:**
  * Purpose: Learn to predict the cluster membership of recipes, which helps in finding similar recipes.
  * Architecture:
    * Input Layer: 50-dimensional input corresponding to the PCA components.
    * Hidden Layers: Two dense layers with 128 and 64 neurons, respectively, using ReLU activation.
    * Output Layer: 25 output neurons with softmax activation to predict cluster probabilities.
  * Compilation: The model is compiled using the Adam optimizer and categorical cross-entropy loss.
* **Model Training:**
  * Data: The PCA-transformed features are used as input (X), and the cluster labels are one-hot encoded as targets (y).
  * Training: The model is trained over 25 epochs with a batch size of 32 and a validation split of 20%.
* **Recipe Recommendation Function:**
  * Purpose: Generate similar recipe recommendations based on a given recipe.
  * Process:
    * Find Target Cluster: Predict the cluster of the given recipe using the neural network.
    * Calculate Similarities: Use cosine similarity to find recipes within the same cluster.
    * Diversification (Optional): Adjust similarity scores to promote diversity in recommendations.
    * Top Recommendations: Select the top N similar recipes, optionally diversified.

This comprehensive approach ensures that the recommendation system is both accurate and efficient, providing users with relevant and varied recipe suggestions based on their preferences and the characteristics of the recipes in the dataset.
