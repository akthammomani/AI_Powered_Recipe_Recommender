# Modeling: 

<center>
    <img src="https://github.com/user-attachments/assets/da4034bd-1a59-4446-a110-370ba92425fb" alt="modeling">
</center>

* **Group 1:** Aktham Almomani, Victor Hsu and Yunus Tezcan
* **Course:** Introduction to Artificial Intelligence (MS-AAI-501) / University Of San Diego
* **Semester:** Summer 2024

## **Introduction**

In this notebook, we will be developing an advanced content-based recommendation system for recipes using various machine learning techniques. The goal is to recommend recipes based on the similarity of their ingredients, categories, diet types, and ratings. We will follow these key steps:

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

## **Dataset**<a id='Dataset'></a>
[Contents](#Contents)

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
