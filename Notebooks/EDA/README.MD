# Exploratory Data Analysis: AI-Powered Recipe Recommender

* **Group 1:** Aktham Almomani, Victor Hsu and Yunus Tezcan
* **Course:** Introduction to Artificial Intelligence (MS-AAI-501) / University Of San Diego
* **Semester:** Summer 2024



<center>
    <img src="https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/7964de28-7421-4718-a3b3-d5ff125d7468" alt="EDA">
</center>

## **Introduction**

Welcome to the Exploratory Data Analysis (EDA) notebook for our AI-Powered Recipe Recommendation Project. This notebook is a crucial phase in our data science process, focused on extracting valuable insights and patterns from our dataset, which will inform the development of our recommendation system.

**In this notebook, we will:**

* Validate the Dataset: Ensure the data is clean, consistent, and ready for analysis.
* Explore Feature Distributions: Analyze the distribution of various features such as recipe categories, diet types, preparation times, and ingredient counts.
* Parse Ingredients: Use Natural Language Processing (NLP) techniques, specifically the SpaCy model, to clean and standardize the ingredients.
* Analyze Recipe Length: Investigate the number of words in recipe directions and how this affects other factors like reviews and ratings.
* Analyze Nutritional Content: Examine the nutritional information across different recipes to categorize them by diet type.
* Correlations Analysis: Investigate correlations between various features such as preparation time, cook time, nutritional values, and recipe popularity.
* Popular Ingredients Analysis: Identify and visualize the most frequently used ingredients in the recipes.

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

## **Exploratory Data Analysis**

**Questions We Aim to Answer:**

* What is the most popular recipe category (most repeated)?
* What is the average number of reviews/ratings?
* What is the average number of reviews/rating count per category?
* What is the average number of reviews/rating count per diet type?
* What is the distribution of reviews/ratings?
* What is the distribution of recipe lengths (number of words in the directions) across all recipes?
* What is the average recipe length for each recipe category?
* What is the average recipe length for each diet type?
* Is there a correlation between recipe length and the number of reviews?
* Is there a correlation between recipe length and average ratings?
* What are the top 10 longest recipes (based on recipe length)?
* What are the top 10 shortest recipes (based on recipe length)?
* Is there a relationship between recipe length and preparation time?
* Is there a relationship between recipe length and cook time?
* Is there a correlation between the number of ingredients and recipe length?
* Is there a correlation between recipe length and the number of calories?
* Is there a correlation between recipe length and the number of servings?
* What is the average prep/cook/total time?
* What are the top 10 most popular ingredients?

By addressing these questions, we aim to gain a comprehensive understanding of the dataset, uncover significant trends and patterns, and prepare the data for developing an effective and user-friendly recipe recommendation system.

### **What is the average number of reviews/rating count per category?**
![What is the average number of reviews rating count per category](https://github.com/user-attachments/assets/53bb6e0f-e3be-4eda-b31a-f52bf3d41968)
**Chart Highlights:**

The chart showcases the top 10 recipe categories based on the average number of reviews. Here are some highlights:

* Pasta and Noodles lead significantly with the highest average number of reviews, indicating strong popularity and engagement in this category.
* Meat and Poultry and Desserts also have high average reviews, reflecting their broad appeal and frequent preparation.
* Categories like Breakfast and Brunch, Fruits and Vegetables, and Drinks also feature prominently, suggesting diverse interest and frequent searches in these areas.
* World Cuisine and Main Dish categories show considerable average reviews, pointing to a wide range of interests in global flavors and main course meals.
* Soups, Stews, and Chili and Bread round out the top 10, highlighting substantial engagement in comfort foods and baking.

This chart helps identify which types of recipes are most engaging to users based on review activity, providing valuable insights for content focus and recommendation strategies.





