# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import cv2
from deepface import DeepFace
import numpy as np
import datetime
import random
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import os
import base64
from io import BytesIO
from PIL import Image

# --- Confirmation Message ---
print("--- Running Updated Server Version ---")

# ----------------------------
# Initialize Flask App
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Data Loading and Preparation
# ----------------------------
def load_data():
    """Load all required data files"""
    df_profit, df_catalog, df_traits, df_mood_prefs = None, None, None, None
    df_pastries = pd.DataFrame() # Default to empty dataframe

    try:
        df_profit = pd.read_csv('profit_margin.csv')
        df_catalog = pd.read_csv('product_catalog.csv')
        df_traits = pd.read_csv('coffee_traits.csv')
        df_mood_prefs = pd.read_csv('mood_preferences.csv')
        print("Successfully loaded core data files.")
    except Exception as e:
        print(f"Error loading a core data file: {e}")
        return None, None, None, None, None

    try:
        df_pastries = pd.read_csv('pastries.csv')
        df_pastries.columns = df_pastries.columns.str.strip()
        df_pastries['Product_Name'] = df_pastries['Product_Name'].astype(str)
        print("Successfully loaded pastries data.")
    except FileNotFoundError:
        print("pastries.csv not found. Continuing without pastry recommendations.")
    except Exception as e:
        print(f"Error loading pastries.csv: {e}")
        print("Continuing without pastry recommendations.")

    # Clean column names
    for df in [df_profit, df_catalog, df_traits, df_mood_prefs]:
        if df is not None:
            df.columns = df.columns.str.strip()

    # Clean product catalog - remove empty rows
    if df_catalog is not None:
        df_catalog = df_catalog.dropna(subset=['product_id'])
        df_catalog = df_catalog[df_catalog['product_id'].notna()]
        df_catalog.loc[df_catalog['product_name'] == 'Spanish_Latte', 'product_name'] = 'Spanish Latte'

    if df_traits is not None and 'hot_or_cold' in df_traits.columns:
        df_traits = df_traits.drop('hot_or_cold', axis=1)

    if df_traits is not None and 'caffeine_level' not in df_traits.columns:
        df_traits['caffeine_level'] = 'medium'
        print("Warning: 'caffeine_level' column not found. Adding a default 'medium' level.")

    # Merge datasets
    df_merged = pd.merge(df_catalog, df_traits, on='product_id')
    df_drinks = pd.merge(df_merged, df_profit, on='product_id')

    if 'calories' in df_drinks:
        df_drinks['calories'] = pd.to_numeric(df_drinks['calories'], errors='coerce')
    else:
        df_drinks['calories'] = 200

    df_drinks['calories'] = df_drinks['calories'].fillna(200)

    df_drinks['mood_tags'] = df_drinks.apply(
        lambda row: [
            'comfort' if row['sweetness'] in ['high', 'medium'] else '',
            'energizing' if row['caffeine_level'] == 'high' else '',
            'relaxing' if row['caffeine_level'] == 'low' else '',
            'indulgent' if 'calories' in row and row['calories'] > 200 else ''
        ], axis=1
    )

    print("Successfully merged all drink data.")
    print(f"Loaded {len(df_drinks)} drinks")

    return df_drinks, df_pastries, df_mood_prefs, df_catalog


# ----------------------------
# Helper Functions
# ----------------------------
def get_sanitized_name(name):
    sanitized_name = name.replace(' ', '_').replace('-', '_')
    while '__' in sanitized_name:
        sanitized_name = sanitized_name.replace('__', '_')
    return sanitized_name

# ----------------------------
# Flask Routes
# ----------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    try:
        in_memory_file = BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        
        image_data = np.frombuffer(in_memory_file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image format or data'}), 400

        results = DeepFace.analyze(img, actions=['emotion', 'age'], enforce_detection=False, silent=True)
        if not results:
            return jsonify({'error': 'No face detected in the image'}), 400

        result = results[0]
        raw_emotions = result['emotion']
        age = result['age']
        
        if not engine:
            return jsonify({'error': 'Application not initialized properly.'}), 500

        enhanced_emotions = EmotionProcessor.enhance_emotion_detection(raw_emotions)
        drink_rec, pastry_rec = engine.recommend_drink(enhanced_emotions, age, context)

        drink_name = drink_rec['product_name'].values[0]
        hot_or_cold = drink_rec['hot_or_cold'].values[0] if 'hot_or_cold' in drink_rec else "hot"
        drink_type_folder = f"{hot_or_cold.lower()}_drinks"
        sanitized_drink_name = get_sanitized_name(drink_name)
        
        drink_image_path = f"static/barns_food_and_drink/{drink_type_folder}/{sanitized_drink_name}.png"
        
        pastry_image_path = None
        if pastry_rec:
            pastry_image_path = f"static/barns_food_and_drink/pastries/{pastry_rec}.png"
            
        display_drink_name = drink_name.replace('_', ' ')
        display_pastry_name = pastry_rec.replace('_', ' ') if pastry_rec else None

        main_category = drink_rec['main_catagory'].values[0] if 'main_catagory' in drink_rec.columns else ''
        if age < 18 and main_category == 'Decaf_Beverages':
            display_drink_name += " (خالي من الكافيين)"
            
        return jsonify({
            'drink_name': display_drink_name,
            'drink_image': drink_image_path,
            'pastry_name': display_pastry_name,
            'pastry_image': pastry_image_path,
            'emotion_data': enhanced_emotions,
            'dominant_emotion': max(enhanced_emotions, key=enhanced_emotions.get).capitalize(),
            'age': int(age)
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An unexpected error occurred.'}), 500

@app.route('/get_price', methods=['POST'])
def get_price():
    product_name = request.json.get('product_name')
    if not product_name:
        return jsonify({'error': 'Product name is required'}), 400

    clean_product_name = product_name.replace(" (خالي من الكافيين)", "").strip().lower()
    
    # Search in drinks catalog
    product_info = df_catalog_global[df_catalog_global['product_name'].str.replace('_', ' ').str.strip().str.lower() == clean_product_name]
    
    if not product_info.empty and 'cost' in product_info.columns:
        return jsonify({'price': float(product_info['cost'].values[0])})

    # Search in pastries catalog
    pastry_info = df_pastries[df_pastries['Product_Name'].str.replace('_', ' ').str.strip().str.lower() == clean_product_name]

    if not pastry_info.empty:
        # CORRECTED: Look for 'Price_SR' column as per user's CSV file
        if 'Price_SR' in pastry_info.columns:
             return jsonify({'price': float(pastry_info['Price_SR'].values[0])})
        elif 'cost' in pastry_info.columns:
            return jsonify({'price': float(pastry_info['cost'].values[0])})
        elif 'Price' in pastry_info.columns:
             return jsonify({'price': float(pastry_info['Price'].values[0])})

    return jsonify({'price': 0.0})


# ----------------------------
# Enhanced Emotion Processing & Other Classes (No changes needed here)
# ----------------------------
class EmotionProcessor:
    EMOTION_HIERARCHY={'angry':0.9,'fear':0.85,'sad':0.8,'disgust':0.75,'surprise':0.7,'happy':0.65,'neutral':0.5}
    EMOTION_GROUPS={'positive':['happy','surprise','relaxed','excited'],'negative':['angry','sad','fear','disgust','anxious','stressed','bored','frustrated'],'neutral':['neutral']}
    COMPLEX_EMOTION_RULES={'anxious':lambda e:e.get('sad',0)>20 and e.get('fear',0)>20,'bored':lambda e:e.get('neutral',0)>50 and sum(e.values())<70,'relaxed':lambda e:e.get('happy',0)>30 and e.get('neutral',0)>40,'stressed':lambda e:e.get('angry',0)>30 and sum(e.values())>80,'excited':lambda e:e.get('happy',0)>30 and e.get('surprise',0)>30,'frustrated':lambda e:e.get('sad',0)>20 and e.get('angry',0)>20}
    @staticmethod
    def enhance_emotion_detection(raw_emotions):
        weighted_emotions={e:s*EmotionProcessor.EMOTION_HIERARCHY.get(e,0.5)for e,s in raw_emotions.items()}
        total=sum(weighted_emotions.values());
        if total==0:return raw_emotions
        enhanced={e:s/total*100 for e,s in weighted_emotions.items()}
        inferred_mood=None
        for mood,rule in EmotionProcessor.COMPLEX_EMOTION_RULES.items():
            if rule(enhanced):inferred_mood=mood;break
        if inferred_mood:
            for emotion in enhanced:enhanced[emotion]=enhanced.get(emotion,0)*0.5
            enhanced[inferred_mood]=50.0
        return enhanced
    @staticmethod
    def get_emotion_group(emotions):
        dominant_emotion=max(emotions,key=emotions.get)
        if dominant_emotion in['bored','anxious','relaxed','stressed','excited','frustrated']:
            if dominant_emotion in['anxious','stressed','bored','frustrated']:return'negative'
            elif dominant_emotion in['relaxed','excited']:return'positive'
        group_scores={g:0 for g in EmotionProcessor.EMOTION_GROUPS}
        for emotion,score in emotions.items():
            for group,emotions_in_group in EmotionProcessor.EMOTION_GROUPS.items():
                if emotion in emotions_in_group:group_scores[group]+=score
        return max(group_scores,key=group_scores.get)

class ContextAnalyzer:
    TIME_CATEGORIES={'morning':(6,12),'afternoon':(12,18),'evening':(18,22),'night':(22,6)}
    SEASON_MONTHS={'winter':[11,12,1,2],'spring':[3,4,5],'summer':[6,7,8],'fall':[9,10]}
    @staticmethod
    def analyze_context():
        now=datetime.now();current_hour=now.hour;current_month=now.month;time_category='night'
        for cat,(start,end) in ContextAnalyzer.TIME_CATEGORIES.items():
            if end>start:
                if start<=current_hour<end:time_category=cat
            else:
                if current_hour>=start or current_hour<end:time_category=cat
        season='summer'
        for s,months in ContextAnalyzer.SEASON_MONTHS.items():
            if current_month in months:season=s
        current_temp=ContextAnalyzer.get_weather()
        weather_condition='Hot'if current_temp>25 else'Cold'
        return{'time_category':time_category,'season':season,'temperature':current_temp,'weather_condition':weather_condition,'weekend':now.weekday()>=5}
    @staticmethod
    def get_weather():
        API_KEY="25f3e394030d70a368078f14d89ecee2";CITY="Jeddah"
        url=f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
        try:
            response=requests.get(url,timeout=3);data=response.json()
            if response.status_code==200 and"main"in data:return data["main"]["temp"]
        except:return 28

class RecommendationEngine:
    def __init__(self,df_drinks,df_pastries,df_mood_prefs):
        self.df_drinks=df_drinks;self.df_pastries=df_pastries;self.last_recommendations=[];self.diversity_factor=0.25
        self.mood_trait_vectors={}
        for _,row in df_mood_prefs.iterrows():
            vector=[]
            for trait in['acidity','aroma','bitterness','sweetness','body']:vector.append(hash(row[f'{trait}_preference'])%100)
            self.mood_trait_vectors[row['mood']]=vector
    def recommend_drink(self,mood_profile,customer_age,context):
        enhanced_emotions=EmotionProcessor.enhance_emotion_detection(mood_profile);emotion_group=EmotionProcessor.get_emotion_group(enhanced_emotions)
        df_scored=self.df_drinks.copy()
        df_scored['mood_score']=df_scored.apply(lambda row:self.calculate_mood_score(row,enhanced_emotions,emotion_group),axis=1)
        df_scored['context_score']=df_scored.apply(lambda row:self.calculate_context_score(row,context,customer_age),axis=1)
        max_profit=df_scored['profit_margine'].max();df_scored['profit_score']=df_scored['profit_margine']/max_profit
        df_scored['diversity_penalty']=df_scored['product_id'].apply(lambda x:0.7 if x in self.last_recommendations[-3:]else 1.0)
        df_scored['final_score']=((df_scored['mood_score']*0.5)+(df_scored['context_score']*0.3)+(df_scored['profit_score']*0.2))*df_scored['diversity_penalty']
        df_scored['final_score']*=(1+self.diversity_factor*(np.random.rand(len(df_scored))-0.5))
        df_sorted=df_scored.sort_values('final_score',ascending=False)
        recommended=df_sorted.head(3).sample(1)
        drink_id=recommended['product_id'].values[0];self.last_recommendations.append(drink_id)
        if len(self.last_recommendations)>10:self.last_recommendations.pop(0)
        pastry_rec=self.recommend_pastry(recommended,emotion_group)
        return recommended,pastry_rec
    def calculate_mood_score(self,drink,emotions,emotion_group):
        mood_similarity=0
        for mood,score in emotions.items():
            if mood in self.mood_trait_vectors and score>0.5:
                drink_vector=[hash(str(drink[t]))%100 for t in['acidity','aroma','bitterness','sweetness','body']]
                similarity=cosine_similarity([self.mood_trait_vectors[mood]],[drink_vector])[0][0]
                mood_similarity+=similarity*(score/100)
        group_boost=0
        if emotion_group=='negative':
            if'comfort'in drink['mood_tags']:group_boost+=1.5
            if drink['sweetness']in['high','medium']:group_boost+=1.0
        elif emotion_group=='positive':
            if'balanced'in drink['mood_tags']:group_boost+=1.2
        else:
            if'balanced'in drink['mood_tags']:group_boost+=1.0
        dominant_emotion=max(emotions,key=emotions.get)
        emotion_rules={'sad':{'bitterness':['low'],'sweetness':['high','medium']},'angry':{'aroma':['strong'],'body':['full']},'happy':{'sweetness':['medium'],'body':['medium']},'fear':{'body':['light']},'bored':{'aroma':['strong'],'sweetness':['high']},'anxious':{'body':['light'],'aroma':['faint']},'relaxed':{'aroma':['balanced'],'sweetness':['mild']},'stressed':{'bitterness':['strong'],'body':['full']},'excited':{'sweetness':['high'],'aroma':['strong']},'frustrated':{'bitterness':['strong'],'sweetness':['low']}}
        emotion_match=0
        if dominant_emotion in emotion_rules:
            for trait,values in emotion_rules[dominant_emotion].items():
                if trait in drink and drink[trait]in values:emotion_match+=1.0
        return mood_similarity+group_boost+emotion_match
    def calculate_context_score(self,drink,context,age):
        score=1.0
        time_rules={'morning':{'aroma':'strong','body':'full'},'afternoon':{'hot_or_cold':'Cold','body':'medium'},'evening':{'body':'light'},'night':{'body':'light'}}
        time_cat=context['time_category']
        if time_cat in time_rules:
            for trait,value in time_rules[time_cat].items():
                if trait in drink and drink[trait]==value:score+=0.8
        if context['weather_condition']=='Hot':
            if'hot_or_cold'in drink and drink['hot_or_cold']=='Cold':score+=1.0
        else:
            if'hot_or_cold'in drink and drink['hot_or_cold']=='Hot':score+=1.0
        if'hot_or_cold'in drink:
            if context['season']=='winter'and drink['hot_or_cold']=='Hot':score+=0.5
            elif context['season']=='summer'and drink['hot_or_cold']=='Cold':score+=0.5
        if context['weekend']:
            if'mood_tags'in drink and'indulgent'in drink['mood_tags']:score+=0.7
        else:
            if'mood_tags'in drink and'energizing'in drink['mood_tags']:score+=0.5
        if age<18:
            if'main_catagory'in drink and drink['main_catagory']=='Decaf_Beverages':score+=2.0
            elif'sweetness'in drink and drink['sweetness']in['high','medium']:score+=1.0
            else:score*=0.1
        return score
    def recommend_pastry(self,drink,emotion_group):
        if self.df_pastries.empty:return""
        pastry_probability=0.7 if emotion_group=='negative'else 0.3
        if random.random()>pastry_probability:return""
        df_scored=self.df_pastries.copy()
        sweetness=drink['sweetness'].values[0]if'sweetness'in drink else'medium'
        bitterness=drink['bitterness'].values[0]if'bitterness'in drink else'medium'
        if bitterness in['high','medium']:df_scored['pairing_score']=df_scored['Product_Name'].apply(lambda x:3.0 if'Chocolate'in str(x)else 2.0 if'Caramel'in str(x)else 1.0)
        elif sweetness in['high','medium']:df_scored['pairing_score']=df_scored['Product_Name'].apply(lambda x:2.5 if any(w in str(x)for w in['Croissant','Bread'])else 1.0)
        else:df_scored['pairing_score']=1.0
        if emotion_group=='negative':df_scored['mood_score']=df_scored['Product_Name'].apply(lambda x:3.0 if any(w in str(x)for w in['Chocolate','Brownie','Cheesecake'])else 1.0)
        else:df_scored['mood_score']=1.0
        df_scored['final_score']=df_scored['pairing_score']*df_scored['mood_score']*np.random.rand(len(df_scored))
        top_pastry=df_scored.nlargest(1,'final_score')
        if not top_pastry.empty:return top_pastry['Product_Name'].values[0]
        else:return""

# ----------------------------
# Initialization
# ----------------------------
engine = None
context = None
df_drinks, df_pastries, df_mood_prefs, df_catalog_global = load_data()

def init_app():
    global engine, context
    if df_drinks is None or df_mood_prefs is None:
        print("Application will not run due to data loading errors.")
        return
    try:
        context = ContextAnalyzer.analyze_context()
        engine = RecommendationEngine(df_drinks, df_pastries, df_mood_prefs)
    except Exception as e:
        print(f"Initialization error: {e}")

if __name__ == '__main__':
    init_app()
    if engine is None:
        print("Engine not initialized. Exiting.")
    else:
        app.run(host="0.0.0.0", port=5000, debug=True)
