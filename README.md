# **AI-Powered Recipe Recommender**

![project_logo](https://github.com/user-attachments/assets/da4034bd-1a59-4446-a110-370ba92425fb)

This project is a part of the Introduction to Artificial Intelligence (AAI-501-02) course in [the Applied Artificial Intelligence Master Program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- **Project Status: Ongoing**

## **Team Members**
* [Aktham Almomani](https://github.com/akthammomani)
* [Yunus Tezcan](https://github.com/fullyuni)
* [Victor Hsu](https://github.com/confooshius)

## **Introduction**

Welcome to the AI-Powered Recipe Recommender! This application aims to enhance your culinary experience by providing personalized recipe recommendations based on various dietary preferences and nutritional needs. Leveraging advanced machine learning and natural language processing (NLP) techniques, our system analyzes a comprehensive dataset of recipes to deliver insightful recommendations and detailed nutritional information.

## **Objective**

The primary objective of this project is to develop an intelligent web application that helps users discover recipes tailored to their individual preferences and dietary requirements. By analyzing recipe data, including ingredients, cooking instructions, and nutritional values, we aim to provide users with:

* Personalized Recipe Recommendations: Suggest recipes based on user-defined criteria such as dietary preferences, nutritional needs, and ingredient availability.
* Nutritional Insights: Offer detailed nutritional breakdowns for each recipe, helping users make informed dietary choices.
* Efficient Search and Filtering: Enable users to search and filter recipes by various attributes, such as preparation time, cooking time, dietary type, and more.
* Data-Driven Insights: Utilize exploratory data analysis (EDA) to uncover trends and patterns within the recipe dataset, providing users with interesting insights about popular recipes, common ingredients, and more.

By integrating sophisticated AI techniques and a user-friendly interface, our application aims to make healthy cooking accessible, enjoyable, and personalized for everyone.

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

## **Methods Used**

* Data Wrangling
* Exploratory Data Analysis (EDA)
* Data Visualization
* Natrual Language Processing (NLP)
* Deep Learning
* Principal component analysis (PCA)
* Unsupervised Machine Learning (K-Means)
* Recommendation System (Advanced Content-Based Recommender)

## **Technologies**

* **Python**: The main programming language used for the project.
* **Streamlit**: For developing and deploying the app using Streamlit Sharing.
* **HTML & CSS**: Web APP personalization.
* **Natrual Language Processing (NLP)**:
  * SpaCy: Features Engineering and Parsing Ingredients features
  * TF-IDF Vectorization: Convert the high_level_ingredients list to a single string and combine relevant features into a single text column.
  * Cosine Similarity: Efficiently ranks recipes within each cluster
* **Principal component analysis (PCA)**: Use PCA to reduce the dimensionality of the TF-IDF vectors (Reduce the complexity of the TF-IDF matrix while preserving essential information).
* **Unsupervised Machine Learning**: Apply KMeans clustering to group similar recipes together to facilitate efficient and relevant recommendations.
* **Deep Learning**: Train a neural network to predict the cluster membership of recipes, which helps in finding similar recipes.

## **Repository Contents**: 
* [Data Wrangling and Pre-Processing Code](https://github.com/akthammomani/AI_Powered_Recipe_Recommender/tree/main/Notebooks/Data_Wrangling_Pre_Processing)
* [Exploratory Data Analysis (EDA) Code](https://github.com/akthammomani/AI_Powered_Recipe_Recommender/tree/main/Notebooks/EDA)
* [Modeling Code](https://github.com/akthammomani/AI_Powered_Recipe_Recommender/tree/main/Notebooks/Modeling)
* [App Development](https://github.com/akthammomani/AI_Powered_Recipe_Recommender/tree/main/App)

## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## **Acknowledgments**

Thank you to **Professor Dave Friesen** for your guidance and support throughout this project/class. Your insights have been greatly appreciated. I also want to extend my gratitude to my team members, whose contributions and collaboration were instrumental in completing this project. Thank you all for your hard work and dedication.
