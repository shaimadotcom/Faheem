# AI Recommendation App

A Flask-based web application that uses AI-powered facial emotion analysis to recommend personalized coffee drinks and pastries from Barns Coffee. The app analyzes user-uploaded photos to detect mood and suggests beverages and treats tailored to their emotional state, time of day, weather, and other contextual factors.

## Features

- **AI-Powered Emotion Detection**: Uses DeepFace library to analyze facial expressions and detect emotions like happy, sad, angry, etc.
- **Personalized Recommendations**: Recommends drinks and pastries based on detected mood, user age, time of day, season, and weather conditions.
- **Context-Aware Suggestions**: Considers factors like weekend status, temperature, and profit margins for optimal recommendations.
- **Mobile-First Design**: Responsive Arabic-language interface optimized for mobile devices.
- **Cart Functionality**: Allows users to add recommended items to a cart and view pricing.
- **Image Upload**: Secure file upload with preview functionality.
- **Real-time Weather Integration**: Fetches current weather data for contextual recommendations.

## Technologies Used

- **Backend**: Python Flask
- **AI/ML**: DeepFace, TensorFlow, scikit-learn, NumPy, Pandas
- **Frontend**: HTML5, CSS3 (Tailwind CSS), JavaScript
- **Image Processing**: OpenCV, Pillow
- **Deployment**: Docker, Gunicorn, Heroku-ready (Procfile included)

## Installation

### Prerequisites
- Python 3.8+
- pip
- Git

### Local Setup

1. **Clone the repository**:


2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional):
   - Create a `.env` file for weather API key:
     ```
     WEATHER_API_KEY=your_openweathermap_api_key
     ```

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Access the app**:
   Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Home Page**: Browse promotional images and navigate using the bottom menu.
2. **Take Photo**: Click the "فهيم" (Faheem) button to learn about the app and upload a photo.
3. **Upload Image**: Select or capture a photo of your face.
4. **Get Recommendations**: The AI analyzes your emotion and provides personalized drink and pastry suggestions.
5. **Add to Cart**: Add recommended items to your cart and proceed to checkout.

## Data Files

The application uses several CSV data files for recommendations:

- `coffee_traits.csv`: Contains coffee characteristics (acidity, aroma, bitterness, sweetness, body, caffeine level)
- `product_catalog.csv`: Product information including names, categories, and pricing
- `mood_preferences.csv`: Maps moods to preferred coffee traits
- `pastries.csv`: Pastry menu with names and prices
- `profit_margin.csv`: Profit margin data for optimization
- `customers.csv`: Customer data (if used for analytics)

## Project Structure

```
Faheem/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── Procfile              # Heroku deployment
├── templates/
│   └── index.html        # Main HTML template
├── static/               # Static assets
│   ├── style.css         # Custom styles
│   ├── logo.png          # Barns Coffee logo
│   ├── entry.png         # Splash screen image
│   ├── spin.gif          # Loading animation
│   ├── video.gif         # Promotional video
│   ├── display_bg.png    # Recommendation background
│   ├── barns_food_and_drink/  # Product images
│   │   ├── cold_drinks/  # Cold drink images
│   │   ├── hot_drinks/   # Hot drink images
│   │   └── pastries/     # Pastry images
│   └── [other images]    # Promotional banners
├── [data files].csv      # Data files for recommendations
└── README.md             # This file
```

## Deployment

### Docker
```bash
docker build -t barns-coffee-app .
docker run -p 5000:5000 barns-coffee-app
```

## API Endpoints

- `GET /`: Main application page
- `POST /upload`: Upload image for emotion analysis and recommendations
- `POST /get_price`: Get pricing information for products


## License

This project is proprietary software for Barns Coffee. All rights reserved.

## Contact

For questions or support, please contact the development team.

---

**Note**: This application requires a valid OpenWeatherMap API key for weather-based recommendations. The AI models may require significant computational resources for emotion detection.