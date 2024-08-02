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

### **What is the average number of reviews/rating count per diet type?**
![What is the average number of reviews rating count per diet type](https://github.com/user-attachments/assets/77526fe3-e596-444f-9296-6de0dca86539)
**Chart Highlights:**

* **Preference for Low Carb and High Protein Diets:** The most reviewed diet types prominently feature low carbohydrate and high protein content. This suggests that users are highly interested in recipes that align with low-carb and high-protein dietary preferences, potentially driven by popular diet trends like keto and Atkins.

* **Combination Diets Gain Popularity:** Diets combining multiple health aspects, such as "Low Carb, Low Fat, High Protein," are notably popular. This indicates a growing interest in recipes that balance multiple nutritional factors, catering to users seeking comprehensive health benefits.

* **General Recipes Still Favored:** Despite the popularity of specialized diets, the "General" category maintains a strong average review count. This shows that while users are exploring specific dietary needs, there is still significant interest in versatile recipes that cater to a broader audience.

* **Health-Conscious Choices:** Diet types focused on reducing fat, sugar, and sodium intake, such as "Low Fat" and "Low Sodium," are well-reviewed. This trend underscores an increasing awareness and preference for health-conscious eating habits among users.

* **Diverse Nutritional Needs:** The variety of diet types in the top 10 list highlights the diverse nutritional needs and interests of the user base. This diversity reflects a market demand for a wide range of recipe options that cater to specific dietary goals and restrictions.

Overall, the high review averages for these diet types suggest that users are actively seeking and engaging with recipes that support specific health and nutritional objectives. This insight can inform future recipe development and content curation to better meet the evolving preferences of health-conscious consumers.

### **What is the average recipe length for each recipe category?**
![What is the average recipe length for each recipe category](https://github.com/user-attachments/assets/c0be519d-1389-46f5-994b-0d1f185bdb8f)
**Chart Highlights:**

The chart highlights the top 10 recipe categories by average recipe length, measured in the number of words in the directions:
* The "holidays-and-events" category stands out significantly, with an average recipe length of 219.12 words, indicating more detailed instructions.
* Following this are "bbq-grilling" and "world-cuisine" with average lengths of 178.00 and 139.36 words, respectively. These longer recipe instructions may suggest more complex or involved cooking processes.
* Other notable categories include "bread," "desserts," and "meat-and-poultry," all averaging over 125 words, reflecting the detailed nature of recipes in these categories.

This analysis helps in understanding the complexity and instructional depth of different recipe categories.

### **What is the average recipe length for each diet type?**
![What is the average recipe length for each diet type](https://github.com/user-attachments/assets/cb3fc09c-6162-4802-b552-98d8b584d0de)
**Chart Highlights:**

The chart shows the top 10 diet types by average recipe length, measured in the number of words in the directions. 
* "High Protein" recipes top the list with an average of 136.19 words, suggesting these recipes may be more detailed and complex.
* Following closely are "General" and "High Protein, Low Sugar" diet types, with average lengths of 134.11 and 132.37 words, respectively. This indicates that recipes focusing on protein content tend to have more elaborate instructions.
* Other notable diet types include "Low Sodium," "High Protein, Low Sodium, Low Sugar," and "Low Carb, High Protein," all averaging over 120 words, reflecting the detailed nature of these diet-specific recipes.

This analysis provides insight into the complexity and instructional depth associated with different dietary preferences.

### **What are the top 10 longest recipes (based on recipe length)?**
![What are the top 10 longest recipes based on recipe length](https://github.com/user-attachments/assets/37e901a5-1705-4712-ba21-15531d12a542)
**Chart Highlights:**
The chart titled "Top 10 Longest Recipes" showcases the recipes with the highest word counts in their instructions:
* All the top recipes are desserts, indicating that dessert recipes tend to be more detailed and elaborate in their preparation steps. This can be due to the precision required in baking and dessert-making, which often involves multiple steps and detailed instructions.
* The longest recipe, "Traditional Layered French Croissants," has 723 words, while the 10th longest, "Coconut-Rum Cake with Lemon Curd and White Chocolate Ganache," has 557 words.

This trend highlights the complexity and detailed nature of dessert recipes compared to other categories.

### **Is there a relationship between recipe length and preparation time?**
![Is there a relationship between recipe length and preparation time](https://github.com/user-attachments/assets/0d104617-e7ac-454d-8941-129bf4c628ba)
**Chart Highlights:**
* Moderate Positive Correlation: The correlation between recipe length and preparation time is 0.33, indicating a moderate positive relationship.
* Dense Cluster at Low Values: Most recipes have shorter preparation times and recipe lengths, with a dense cluster below 200 words and 100 minutes.
* Outliers: There are some outliers with very high preparation times, exceeding 600 minutes, and some recipes with lengths over 500 words.
* Trend Visibility: While the overall trend shows a positive correlation, the scatter plot reveals significant variability, especially for recipes with longer lengths.

This visualization helps to understand how preparation time generally increases with recipe length, but the relationship is not very strong, indicating other factors might also influence preparation time.

### **Is there a relationship between recipe length and cook time?**
![Is there a relationship between recipe length and cook time](https://github.com/user-attachments/assets/2def234c-f815-47c3-b219-ead9214b3f77)
**Chart highlights:**
* The correlation between recipe length and cook time is very weak (0.02), suggesting no significant relationship between the two variables. This differs from the previous analysis, where there was a moderate positive correlation between recipe length and preparation time.
* This discrepancy could indicate that the complexity or length of a recipe (in terms of instructions) has more to do with preparation activities rather than cooking duration. It might imply that longer recipes involve more detailed preparation steps, but the cooking time does not necessarily increase proportionally.

This insight can be valuable for understanding the nature of recipes and how users might plan their cooking activities.

### **Is there a correlation between the number of ingredients and recipe length?**
![Is there a correlation between the number of ingredients and recipe length](https://github.com/user-attachments/assets/7ef9f11f-c94e-452b-a668-2c4be5d7de92)
**Chart Highlights:**

* Moderate Positive Correlation: There is a moderate positive correlation of 0.45 between the number of ingredients and recipe length. This indicates that recipes with more ingredients tend to have longer instructions.
* Recipe Complexity: Recipes with a higher ingredient count, often indicative of more complex or detailed recipes, generally have a longer recipe length.
* Variation Across Recipes: There is a significant variation in recipe lengths even for recipes with a similar number of ingredients, suggesting that the way instructions are written can vary widely.

### **Is there a correlation between recipe length and the number of calories?**
![Is there a correlation between recipe length and the number of calories](https://github.com/user-attachments/assets/7d78bbb3-22a6-41c2-ac39-2f12b69ba1b7)
**Chart Highlights:**
* Positive Correlation: The scatter plot shows a moderate positive correlation (0.23) between recipe length and the number of calories. This suggests that recipes with more steps or longer instructions tend to have higher calorie counts.
* Dense Cluster: A dense cluster of points is observed at the lower left of the plot, indicating that most recipes have shorter lengths and lower calorie counts.
* Outliers: There are some outliers with high calorie counts and longer recipe lengths, suggesting that a few recipes with detailed instructions contain significantly more calories.
* Overall Trend: As the number of words in a recipe increases, there is a slight trend towards higher calorie content, but this trend is not very strong.

### **What are the top 10 most popular ingredients?**
![What are the top 10 most popular ingredients](https://github.com/user-attachments/assets/7191cfc6-a368-4f25-b6f8-20826488ead7)
**Chart Highlights:**
* Salt is the most popular ingredient, appearing in 13,522 recipes.
* Black pepper follows as the second most common ingredient, with 7,652 occurrences.
* Butter is third, used in 6,381 recipes, highlighting its essential role in cooking and baking.
* All-purpose flour and white sugar are also frequently used, with counts of 6,072 and 6,002, respectively, indicating their importance in various recipes.
* Onion and olive oil are notable ingredients in savory dishes, found in 5,114 and 4,968 recipes.
* Water, while commonly available, appears in 4,725 recipes, underscoring its utility in cooking processes.
* Eggs and cloves complete the top 10 list, with 3,896 and 3,720 occurrences, showing their significance in both baking and savory dishes.

These ingredients are staples in many kitchens and form the foundation of a wide variety of recipes.











