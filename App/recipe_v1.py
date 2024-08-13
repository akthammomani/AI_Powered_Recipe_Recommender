import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

icon = Image.open("chef.jpg")
st.set_page_config(layout='centered', page_title='AI-Powered Recipe Recommender', page_icon=icon)

# Let's upload the Painted Ladies image:
image = Image.open("project_logo.JPG")
# Let's specify which column, fix its width, and let's give this image a caption:
st.image(Image.open("project_logo.JPG"), use_column_width=True)


# Load the saved models and components
with open('recipe_recommendation_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('pca_model.pkl', 'rb') as file:
    pca = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

# Use the cached function to load the data
df = load_data('all_recipes_final_df_v2.csv')

# Update the columns to reflect grams with daily percentage
df['Carbohydrates g(Daily %)'] = df.apply(lambda x: f"{x['carbohydrates_g']}g ({x['carbohydrates_g_dv_perc']}%)", axis=1)
df['Sugars g(Daily %)'] = df.apply(lambda x: f"{x['sugars_g']}g ({x['sugars_g_dv_perc']}%)", axis=1)
df['Fat g(Daily %)'] = df.apply(lambda x: f"{x['fat_g']}g ({x['fat_g_dv_perc']}%)", axis=1)
df['Protein g(Daily %)'] = df.apply(lambda x: f"{x['protein_g']}g ({x['protein_g_dv_perc']}%)", axis=1)


# Transform the combined features using the loaded TF-IDF vectorizer and PCA model
tfidf_matrix = tfidf.transform(df['combined_features'])  # Use transform instead of fit_transform
tfidf_pca = pca.transform(tfidf_matrix.toarray())  # Use transform instead of fit_transform



# Rename the columns to user-friendly names
friendly_names = {
    'name': 'Recipe Name',
    'category': 'Category',
    'calories': 'Calories (kcal)',
    'servings': 'Servings',
    'Carbohydrates g(Daily %)': 'Carbohydrates g(Daily %)',
    'Sugars g(Daily %)': 'Sugars g(Daily %)',
    'Fat g(Daily %)': 'Fat g(Daily %)',
    'Protein g(Daily %)': 'Protein g(Daily %)',
    'cook': 'Cook Time (minutes)',
    'rating': 'Rating',
    'rating_count': 'Rating Count',
    'diet_type' : 'Diet Type',
    'ingredients': 'Ingredients',
    'directions': 'Directions'
            }
# Function to get similar recipes
def get_similar_recipes(recipe_name, top_n=5, diversify=False, diversity_factor=0.1):
    target_index = df[df['name'] == recipe_name].index[0]
    target_features = tfidf.transform([df['combined_features'].iloc[target_index]])
    target_features_pca = pca.transform(target_features.toarray())
    target_cluster = model.predict(target_features_pca).argmax()
    cluster_indices = df[df['cluster'] == target_cluster].index
    similarities = cosine_similarity(target_features_pca, tfidf_pca[cluster_indices]).flatten()
    weighted_similarities = similarities * df.loc[cluster_indices, 'rating']
    
    if diversify:
        diversified_scores = weighted_similarities * (1 - diversity_factor * np.arange(len(weighted_similarities)))
        similar_indices = cluster_indices[np.argsort(diversified_scores)[-top_n:][::-1]]
    else:
        similar_indices = cluster_indices[np.argsort(weighted_similarities)[-top_n:][::-1]]
    
    # Retrieve similar recipes and sort them by rating_count and rating
    similar_recipes = df.iloc[similar_indices]
    similar_recipes_sorted = similar_recipes.sort_values(by=['rating_count', 'rating'], ascending=False)
    
    # Select only the desired columns
    selected_columns = ['name', 'category', 'ingredients', 'directions','rating', 'rating_count', 'diet_type','calories', 'servings', 'Carbohydrates g(Daily %)', 'Sugars g(Daily %)', 'Fat g(Daily %)', 'Protein g(Daily %)', 'cook']
    selected_recipes = similar_recipes_sorted[selected_columns].head(top_n)
    
    
    return selected_recipes.rename(columns=friendly_names)

# Function to filter recipes by servings
def filter_by_servings(servings):
    if servings == "one":
        return df[df['servings'] == 1]
    elif servings == "two":
        return df[df['servings'] == 2]
    elif servings == "crowd":
        return df[df['servings'] >= 5]
    else:
        return pd.DataFrame()

# Function to filter recipes by name
def filter_by_recipe_name(name):
    return df[df['name'].str.contains(name, case=False, na=False)]


# Function to filter and sort recipes
def filter_and_sort_by_recipe_name(name):
    results = filter_by_recipe_name(name)
    return results.sort_values(by=['rating_count','rating'], ascending=False)

def autocomplete_suggestions(user_input, df, max_suggestions=5):
    # Filter recipe names that contain the user input
    filtered_df = df[df['name'].str.contains(user_input, case=False, na=False)]
    
    # Sort by rating_count to prioritize popular recipes
    sorted_df = filtered_df.sort_values(by='rating_count', ascending=False)
    
    # Return the top `max_suggestions` recipe names
    return sorted_df['name'].head(max_suggestions).tolist()

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

#   - Recommendations are automatically diversified to offer you a broader variety of results.

st.markdown(
    """
    ## Introduction
    
    Are you in search of the perfect recipe? Look no further! Our Recipe Recommendation App is designed to help you discover delicious recipes tailored to your preferences. Whether you're searching for something specific or exploring new dishes, this app offers a variety of features to enhance your culinary journey.
    
    #### Search Options:
    
    **1) Personalized Recommendations:**
    - Simply enter the name of a recipe, and the app will suggest similar recipes tailored to your preferences.
    - Autocomplete suggestions guide you to the exact recipe name you're looking for.
    - Our Recommender system is based on Rating, Category, Diet Type, and Ingredients.
    
    **2) Popular Searches:**
    - Quickly access popular and trending recipes.
    - Search by common keywords like "Chicken," "Pancakes," or "Lasagna."
    - Filter by serving size for tailored results.
    
    **3) Custom Search:**
    - Select recipes based on categories like "Main Dish," "Desserts," or "World Cuisine."
    - Filter by diet type, such as "Low Carb" or "High Protein."
    - Choose recipes based on serving size or cooking time, perfect for specific meal planning needs.
    """
)

st.write('---')

st.markdown(
    """
    ## Find Your Perfect Recipe
    """
)
option = st.selectbox(
    'How would you like to search for recipes?',
    ('Personalized Recommendations', 'Popular Searches', 'Custom Search')
)


if option == 'Personalized Recommendations':
    #st.header('Get Recommendations')

    st.write(
"""
    ### **Get Recommendations**
""") 
    
    # Input field with suggestions
    user_input = st.text_input('Enter a Recipe Name')
    
    suggestions = []
    if user_input:
        suggestions = autocomplete_suggestions(user_input, df)

    selected_recipe = None
    if suggestions:
        st.write("Did you mean:")
        for suggestion in suggestions:
            if st.button(suggestion):
                selected_recipe = suggestion
                break  # Exit the loop once a selection is made

    if selected_recipe:
        #top_n = st.slider('Number of Recommendations', 1, 10, 5)
        #diversify = st.checkbox('Diversify Recommendations')
        
        similar_recipes = get_similar_recipes(selected_recipe, top_n=10, diversify=0)
        st.write(f"Top {10} recommendations for '{selected_recipe}':")
        st.dataframe(similar_recipes.reset_index(drop=True).reset_index(drop=False).rename(columns={'index': 'Rank'}).assign(Rank=lambda x: x.index + 1), hide_index=True)

        st.write("#### Learn More")
        st.markdown("[![](https://img.shields.io/badge/GitHub%20-Recipes%20Recommender-informational)](https://github.com/akthammomani/AI_Powered_Recipe_Recommender/tree/main/Notebooks/Modeling)")
    
    elif user_input and not selected_recipe:
        if st.button('Get Recommendations'):
            st.warning('No matching recipes found. Please try again.')

elif option == 'Popular Searches':
    #st.header('Popular Searches')

    st.write(
"""
    ### **Popular Searches**
""") 
    results = pd.DataFrame()  # Initialize results to avoid NameError

    # Search bar for direct recipe name search
    search_query = st.text_input("What would you like to cook?")
    if search_query:
        results = filter_by_recipe_name(search_query)
        results = results.sort_values(by=['rating_count','rating'], ascending=False)

    def filter_and_sort_by_servings(servings):
        filtered_results = filter_by_servings(servings)
        return filtered_results.sort_values(by=['rating_count','rating'], ascending=False)

    def filter_and_sort_by_recipe_name(name):
        filtered_results = filter_by_recipe_name(name)
        return filtered_results.sort_values(by=['rating_count', 'rating'], ascending=False)

    # Search by Servings
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Cooking for one'):
            results = filter_and_sort_by_servings("one")
    #with col2:
        if st.button('Cooking for two'):
            results = filter_and_sort_by_servings("two")
    #with col3:
        if st.button('Cooking for crowd'):
            results = filter_and_sort_by_servings("crowd")  
    # Search by Recipe Name
    #col1, col2, col3 = st.columns(3)
    with col2:
        if st.button('Fish'):
            results = filter_and_sort_by_recipe_name("fish")
        if st.button('Beef'):
            results = filter_and_sort_by_recipe_name("Beef")
        if st.button('Chicken'):
            results = filter_and_sort_by_recipe_name("Chicken")            
    with col3:
        if st.button('Lasagna'):
            results = filter_and_sort_by_recipe_name("Lasagna")
        if st.button('Pancakes'):
            results = filter_and_sort_by_recipe_name("Pancakes")
        if st.button('Banana Bread'):
            results = filter_and_sort_by_recipe_name("Banana Bread")    

    if not results.empty:
        st.write("Showing results for your search:")
        #st.write(results)
            # Select only the desired columns
        selected_columns = ['name', 'category', 'ingredients', 'directions','rating', 'rating_count', 'diet_type','calories', 'servings', 'Carbohydrates g(Daily %)', 'Sugars g(Daily %)', 'Fat g(Daily %)', 'Protein g(Daily %)', 'cook']
        results = results[selected_columns].head(10)
        results = results.rename(columns=friendly_names)
        st.dataframe(results.reset_index(drop=True).reset_index(drop=False).rename(columns={'index': 'Rank'}).assign(Rank=lambda x: x.index + 1), hide_index=True)


elif option == 'Custom Search':
    #st.header('Search Recipes')

    st.write(
"""
    ### **Search Recipes**
""") 
    
    category = st.selectbox('Category', [
        'appetizers-and-snacks', 'desserts', 'world-cuisine', 'main-dish', 
        'side-dish', 'bread', 'soups-stews-and-chili', 'meat-and-poultry', 
        'salad', 'seafood', 'breakfast-and-brunch', 'trusted-brands-recipes-and-tips', 
        'everyday-cooking', 'fruits-and-vegetables', 'pasta-and-noodles', 
        'drinks', 'holidays-and-events', 'bbq-grilling'
    ])
    
    diet_type = st.selectbox('Diet Type', [
        'General', 'High Protein', 'Low Carb, Low Sugar', 'Low Carb, High Protein, Low Sugar',
        'High Protein, Low Sugar', 'Low Fat', 'Low Sugar', 'Low Carb, Low Fat, Low Sugar',
        'Low Fat, Low Sugar', 'Low Carb, Low Fat, Low Sodium, Low Sugar', 'Low Fat, Low Sodium',
        'Low Carb, Low Fat, Low Sodium', 'Low Sodium', 'Low Carb, High Protein', 'Low Carb',
        'Low Carb, Low Fat', 'Low Carb, Low Fat, High Protein, Low Sugar', 'Low Fat, High Protein',
        'Low Carb, Low Sodium, Low Sugar', 'Low Fat, Low Sodium, Low Sugar',
        'Low Fat, High Protein, Low Sugar', 'Low Carb, Low Sodium', 'Low Carb, Low Fat, High Protein',
        'Low Sodium, Low Sugar', 'Low Carb, High Protein, Low Sodium, Low Sugar',
        'Low Carb, Low Fat, High Protein, Low Sodium, Low Sugar', 'High Protein, Low Sodium, Low Sugar',
        'High Protein, Low Sodium', 'Low Carb, High Protein, Low Sodium', 'Low Fat, High Protein, Low Sodium',
        'Low Fat, High Protein, Low Sodium, Low Sugar', 'Low Carb, Low Fat, High Protein, Low Sodium'
    ])
    
    ingredients = st.text_input('Ingredients (comma-separated)')
    
    serving_one = st.checkbox('Cooking for One')
    serving_two = st.checkbox('Cooking for Two')
    serving_crowd = st.checkbox('Cooking for a Crowd')
    
    quick_and_easy = st.checkbox('Quick and Easy Recipes')
    
    #num_results = st.slider('Number of Results to Show', 1, 50, 10)
    
    if st.button('Search'):
        query_parts = []
        if category:
            query_parts.append(f'category == "{category}"')
        
        if diet_type and diet_type != 'General':  # Apply diet type filter if selected
            query_parts.append(f'diet_type == "{diet_type}"')
        
        if ingredients:
            ingredients_list = ingredients.split(',')
            ingredients_query = ' & '.join([f'high_level_ingredients_str.str.contains("{ingredient.strip()}")' for ingredient in ingredients_list])
            query_parts.append(ingredients_query)
        
        # Apply serving size filter
        if serving_one:
            query_parts.append('servings == 1')
        if serving_two:
            query_parts.append('servings == 2')
        if serving_crowd:
            query_parts.append('servings >= 5')
        
        # Apply quick and easy filter
        if quick_and_easy:
            query_parts.append('cook_time_mins <= 15')
        
        query = " & ".join(query_parts)
        
        # Print the query for debugging purposes
        #st.write(f"Query: {query}")
        
        if query:
            # Filter the DataFrame
            filtered_recipes = df.query(query)
            
            # No need to sort again since df is already sorted
            st.write(f"Showing top {10} results for the given filters:")
            #st.write(filtered_recipes.head(num_results))
            selected_columns = ['name', 'category', 'ingredients', 'directions','rating', 'rating_count', 'diet_type','calories', 'servings', 'Carbohydrates g(Daily %)', 'Sugars g(Daily %)', 'Fat g(Daily %)', 'Protein g(Daily %)', 'cook']
            filtered_recipes = filtered_recipes.sort_values(by=['rating_count', 'rating'], ascending=False)
            filtered_recipes = filtered_recipes[selected_columns]
            filtered_recipes = filtered_recipes.rename(columns=friendly_names)
            st.dataframe(filtered_recipes.reset_index(drop=True).reset_index(drop=False).rename(columns={'index': 'Rank'}).assign(Rank=lambda x: x.index + 1), hide_index=True)
        else:
            st.warning('Please enter at least one filter.')

st.write('---')

#null9_0, row9_1, row9_2 = st.columns((0, 5, 0.05))
with st.expander("Leave Us a Comment or Question"):
    contact_form = """
        <form action=https://formsubmit.co/aktham.momani81@gmail.com method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here"></textarea>
            <button type="submit">Send</button>
        </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    local_css("style.css")

st.write("""
    ### Contacts
    [![](https://img.shields.io/badge/GitHub-Recipes%20Recommender-informational)](https://github.com/akthammomani/AI_Powered_Recipe_Recommender/)
    [![](https://img.shields.io/badge/Open-Issue-informational)](https://github.com/akthammomani/AI_Powered_Recipe_Recommender/issues)
    [![MAIL Badge](https://img.shields.io/badge/-aktham.momani81@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:aktham.momani81@gmail.com)](mailto:aktham.momani81@gmail.com)
    ###### © All rights reserved.
    """)
