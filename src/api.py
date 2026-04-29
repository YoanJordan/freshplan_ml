from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import sys
sys.path.append('/Users/yoanjordan/Desktop/freshplan_ml')

app = FastAPI(title="FreshPlan ML API")

# Load model and data at startup
with open('/Users/yoanjordan/Desktop/freshplan_ml/models/model.pkl', 'rb') as f:
    model_data = pickle.load(f)

recipes = pd.read_csv('/Users/yoanjordan/Desktop/freshplan_ml/data/clean/recipes_clean.csv')

user_factors = model_data['user_factors']
recipe_factors = model_data['recipe_factors']
user_to_idx = model_data['user_to_idx']
recipe_to_idx = model_data['recipe_to_idx']
user_ids = model_data['user_ids']
recipe_ids = model_data['recipe_ids']


# Data models
class RecommendationRequest(BaseModel):
    user_id: int
    fridge_ingredients: List[str]
    allergens: List[str] = []
    history_count: int = 0
    n_recommendations: int = 10


class Recipe(BaseModel):
    recipe_id: int
    recipe_name: str
    score: float


class RecommendationResponse(BaseModel):
    mode: str
    recipes: List[Recipe]


# ML functions
def content_score(fridge_ingredients, recipe_ingredients_str):
    fridge = [i.lower().strip() for i in fridge_ingredients]
    recipe = [i.lower().strip() for i in recipe_ingredients_str.split('^')]
    if len(recipe) == 0:
        return 0
    matches = sum(1 for ingredient in recipe
                  if any(f in ingredient or ingredient in f for f in fridge))
    return matches / len(recipe)


def hard_filter(recipes_df, allergens):
    if not allergens:
        return recipes_df
    def contains_allergen(ingredients_str):
        ingredients = str(ingredients_str).lower()
        return any(a.lower() in ingredients for a in allergens)
    return recipes_df[~recipes_df['ingredients'].apply(contains_allergen)]


def final_score(u_idx, recipe_id, fridge_ingredients,
                weight_content=0.6, weight_preference=0.4):
    if recipe_id in recipe_to_idx:
        r_idx = recipe_to_idx[recipe_id]
        pref_score = float(user_factors[u_idx] @ recipe_factors[r_idx])
        pref_score = max(0, min(1, pref_score))
    else:
        pref_score = 0
    recipe_row = recipes[recipes['recipe_id'] == recipe_id]
    if recipe_row.empty:
        return 0
    c_score = content_score(fridge_ingredients, recipe_row.iloc[0]['ingredients'])
    return round(weight_content * c_score + weight_preference * pref_score, 4)


# API routes
@app.get("/")
def root():
    return {"message": "FreshPlan ML API is running!"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    # Apply hard filter first
    available_recipes = hard_filter(recipes, request.allergens)

    # Set weights based on user history
    if request.history_count < 5:
        wc, wp, mode = 0.9, 0.1, "COLD-START"
    else:
        wc, wp, mode = 0.6, 0.4, "NORMAL"

    # Get user index
    u_idx = user_to_idx.get(request.user_id, 0)

    # Compute scores for all recipes
    results = []
    for _, row in available_recipes.iterrows():
        score = final_score(u_idx, row['recipe_id'],
                           request.fridge_ingredients, wc, wp)
        results.append({
            'recipe_id': int(row['recipe_id']),
            'recipe_name': row['recipe_name'],
            'score': score
        })

    # Return top N results
    top_n = sorted(results, key=lambda x: x['score'], reverse=True)[:request.n_recommendations]

    return RecommendationResponse(mode=mode, recipes=top_n)

