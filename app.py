from flask import Flask, render_template, request
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Home route for the webpage (GET request)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle sentiment prediction (POST request)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the review text from the form
        review_text = request.form['review']
        
        # Transform the input review using the vectorizer
        review_vectorized = vectorizer.transform([review_text])
        
        # Predict sentiment using the model
        prediction = model.predict(review_vectorized)
        
        # Convert prediction to readable format
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        
        return render_template('index.html', prediction_text=f"The review is: {sentiment}")

if __name__ == "__main__":
    app.run(debug=True)
