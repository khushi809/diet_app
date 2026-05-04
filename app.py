from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)

print("App is starting...")

BASE = os.path.dirname(__file__)

# ── Database setup ─────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(os.path.join(BASE, "users.db"))
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_inputs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age       INTEGER,
            weight    REAL,
            height    REAL,
            goal      TEXT,
            diet      TEXT,
            calories  INTEGER,
            protein   REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(age, weight, height, goal, diet, calories, protein):
    conn = sqlite3.connect(os.path.join(BASE, "users.db"))
    c = conn.cursor()
    c.execute('''
        INSERT INTO user_inputs (timestamp, age, weight, height, goal, diet, calories, protein)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), age, weight, height, goal, diet, calories, protein))
    conn.commit()
    conn.close()

init_db()
print("Database ready.")

# ── Load ML model ──────────────────────────────────────────────────────────
with open(os.path.join(BASE, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

df_raw = pd.read_csv(os.path.join(BASE, "Indian_Food_Nutrition_Processed.csv"))
df_raw = df_raw.rename(columns={
    'Dish Name': 'Food', 'Calories (kcal)': 'Calories',
    'Carbohydrates (g)': 'Carbs', 'Protein (g)': 'Protein', 'Fats (g)': 'Fat'
})
df = df_raw[['Food','Calories','Protein','Carbs','Fat']].dropna().drop_duplicates().reset_index(drop=True)

# Remove anything that is not a proper meal
EXCLUDE = (
    r'chutney|pickle|papad|murabba|squash|sherbet|syrup|soda|sharbat|panna|'
    r'tea|coffee|lassi|buttermilk|drink|juice|water|soup|consomme|broth|'
    r'powder|masala|spice|chaat masala|gun powder|'
    r'candy|toffee|chocolate|mithai|ladoo|barfi|halwa|kheer|pudding|custard|'
    r'ice cream|kulfi|sorbet|mousse|'
    r'jam|jelly|honey|preserve|murabba|'
    r'ghee|butter|oil|cream|'
    r'raita|dip|sauce|ketchup|'
    r'papad|fryum|chips|namkeen|'
    r'paan|supari|gutka|tobacco'
)
df = df[~df['Food'].str.contains(EXCLUDE, case=False, na=False)].reset_index(drop=True)

# Only keep foods with meaningful calories (at least 80 kcal)
df = df[df['Calories'] >= 80].reset_index(drop=True)

# XGBoost predicts protein for every food at startup
features = scaler.transform(df[['Calories','Carbs','Fat']].values)
df['ML_Predicted_Protein'] = model.predict(features)
print(f"XGBoost predictions ready for {len(df)} foods.")

NON_VEG = r'chicken|egg|fish|mutton|prawn|shrimp|crab|lobster|pork|beef|lamb|tuna|salmon|sardine|meat|bacon|sausage'

def filter_diet(df_in, diet_type):
    if diet_type == "veg":
        return df_in[~df_in['Food'].str.contains(NON_VEG, case=False, na=False)]
    return df_in

def recommend_food(target_protein, used_foods, df_filtered, n=2):
    temp = df_filtered[~df_filtered['Food'].isin(used_foods)].copy()
    temp['ml_diff'] = abs(temp['ML_Predicted_Protein'] - target_protein)
    pool = temp.sort_values('ml_diff').head(15)
    top  = pool.sample(n=min(n, len(pool)))
    return top[['Food','Protein','ML_Predicted_Protein','Calories','Carbs','Fat']].to_dict(orient='records')

def calculate_calories(weight, goal):
    base = 24 * weight
    if goal == "loss":   return round(base * 0.85)
    elif goal == "gain": return round(base * 1.15)
    else:                return round(base)

def calculate_protein(weight, goal):
    return round({"loss": 1.2, "maintain": 1.0, "gain": 1.6}.get(goal, 1.0) * weight, 1)

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data      = request.json
    goal      = data.get("goal", "maintain")
    diet_type = data.get("diet", "veg")
    weight    = float(data.get("weight", 60))
    height    = float(data.get("height", 165))
    age       = int(data.get("age", 25))

    calories          = calculate_calories(weight, goal)
    protein_needed    = calculate_protein(weight, goal)
    breakfast_protein = 0.30 * protein_needed
    lunch_protein     = 0.40 * protein_needed
    dinner_protein    = 0.30 * protein_needed

    df_filtered = filter_diet(df, diet_type)

    used = []
    breakfast = recommend_food(breakfast_protein, used, df_filtered)
    used += [f['Food'] for f in breakfast]
    lunch = recommend_food(lunch_protein, used, df_filtered)
    used += [f['Food'] for f in lunch]
    dinner = recommend_food(dinner_protein, used, df_filtered)

    # Save user input to database
    save_to_db(age, weight, height, goal, diet_type, calories, protein_needed)

    return jsonify({
        "status": "ok",
        "inputs": {"goal": goal, "diet": diet_type, "weight": weight, "height": height, "age": age},
        "nutrition": {"calories": calories, "protein": protein_needed,
            "breakdown": {
                "breakfast": round(breakfast_protein, 1),
                "lunch":     round(lunch_protein, 1),
                "dinner":    round(dinner_protein, 1)
            }
        },
        "meals": {"breakfast": breakfast, "lunch": lunch, "dinner": dinner}
    })


import sqlite3
import pandas as pd
from flask import render_template


# ── Admin route ────────────────────────────────────────────────────────────
@app.route("/admin")
def admin():
    conn = sqlite3.connect(os.path.join(BASE, "users.db"))
    df_users = pd.read_sql_query("SELECT * FROM user_inputs ORDER BY id DESC", conn)
    conn.close()
    return str(df_users)

import os

print("Starting Flask server...")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # IMPORTANT
    app.run(host="0.0.0.0", port=port)

    