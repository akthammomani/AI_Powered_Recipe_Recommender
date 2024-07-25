# **Data Wrangling and Processing Stage**

<center>
    <img src="https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/2cab2215-ce7f-4951-a43a-02b88a5b9fa9" alt="wrnagling">
</center>

## **Introduction**

This project involves developing a comprehensive data wrangling and pre-processing pipeline for a recipe dataset. The objective is to clean, parse, and enhance the dataset to facilitate further analysis and model building for an AI-powered recipe recommender system. Key steps include handling missing values, standardizing ingredient formats using NLP techniques, verifying cooking times, and creating new features such as diet type and recipe length.

* **Data Cleaning:**
  * Handled Missing Values: Filled missing nutritional values with zeros to maintain data consistency.
  * Removed High Missing Data Percentage Columns: Eliminated columns with a high percentage of missing values to ensure data quality.
  * Eliminated Duplicates: Removed duplicate entries in recipe names to prevent redundancy.
* **Feature Engineering:**
  * Parsed and Cleaned Ingredients: Utilized NLP with SpaCy to parse and standardize the 'ingredients' column, ensuring uniform ingredient formats.
  * Verified Total Cooking Times: Ensured accuracy by summing preparation and cooking times and comparing them with the total time provided.
  * Created 'Diet Type' Column: Categorized recipes based on their nutritional content, such as low carb, low fat, high protein, low sodium, and low sugar.
  * Added 'Recipe Length' Feature: Calculated the number of words in the directions column to analyze recipe complexity and verbosity.

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

## **Cleaning and Parsing Ingredients for Standardized Analysis**

Developed a custom function to 

 cleans and parses the ingredient strings by:

* Splitting: It divides the ingredient string into a list using semicolons (;) as delimiters.
* Trimming: It removes any leading or trailing whitespace from each ingredient in the list.

```python
# Function to clean and parse ingredients
def parse_ingredients(ingredient_str):
    # Split the string by the delimiter ';'
    ingredients_list = ingredient_str.split(';')
    # Clean up any leading/trailing whitespace and return the list
    return [ingredient.strip() for ingredient in ingredients_list]
```

## **Utilizing spaCy (NLP) for Ingredient Extraction and Cleaning**

In this section, we utilize the **spaCy** library for natural language processing to clean and extract high-level ingredients from recipe text data. The code first removes unwanted characters and patterns from the ingredient descriptions. Then, it extracts meaningful high-level ingredient names by filtering out common stop words and measurement terms. This process helps to standardize the ingredient data for better analysis and modeling.

The final dataset includes a new column, **high_level_ingredients**, which contains cleaned and high-level ingredient names, and an **ingredient_count** column, which represents the number of high-level ingredients in each recipe. This approach enhances the quality and consistency of ingredient data, making it more useful for downstream analysis and machine learning tasks.

```python
import spacy
nlp = spacy.load("en_core_web_sm-3.1.0")

# List of unwanted characters and patterns to exclude
unwanted_patterns = [
    r'\u2009', r'/', r'inch', r'â…›', r'â…”', r'Â®"', r'®', r'\)', r'\(', r'%', r'V8', r'V8®', r'™', r'®', r'\'', r'"'
]

# Function to clean and trim ingredient text
def clean_ingredient_text(text):
    text = text.strip()
    for pattern in unwanted_patterns:
        text = re.sub(pattern, '', text)
    return text

# Function to extract high-level ingredients using spaCy:
def extract_high_level_ingredients(parsed_ingredients):
    high_level_ingredients = []
    
    # Custom stop words list to filter out non-ingredient words:
    stop_words = set([
        'cup', 'cups', 'teaspoon', 'teaspoons', 'tablespoon', 'tablespoons', 'ounce', 'ounces',
        'pound', 'pounds', 'quart', 'quarts', 'pinch', 'dash', 'taste', 'large', 'small', 'medium',
        'divided', 'minced', 'sliced', 'diced', 'chopped', 'ground', 'freshly', 'prepared', 'cut',
        'into', 'strips', 'halves', 'cubes', 'to', 'box', 'spoon', 'spoons', 'optional'
    ])
    
    for ingredient in parsed_ingredients:
        # Clean and trim ingredient text
        ingredient = clean_ingredient_text(ingredient)
        
        # Remove numbers, fractional numbers, and measurement words
        ingredient = re.sub(r'\d*\s*[\d¼½¾⅓⅔⅛⅜⅝⅞]+\s*', '', ingredient)  # Remove any numbers or fractional numbers
        ingredient = re.sub(r'\b(?:' + '|'.join(stop_words) + r')\b', '', ingredient, flags=re.IGNORECASE)  # Remove stop words

        # Remove any remaining unwanted characters and patterns
        ingredient = clean_ingredient_text(ingredient)
        
        doc = nlp(ingredient)
        for chunk in doc.noun_chunks:
            filtered_words = [token.text for token in chunk if token.text.lower() not in stop_words and not token.is_digit]
            if filtered_words:
                high_level_ingredients.append(' '.join(filtered_words).strip())
    
    return list(set(high_level_ingredients))  # Remove duplicates

# Apply the function to the 'parsed_ingredients' column:
df['high_level_ingredients'] = df['parsed_ingredients'].apply(extract_high_level_ingredients)

# Create the ingredient_count column:
df['ingredient_count'] = df['high_level_ingredients'].apply(len)

df.head()
```

