from flask import Flask, render_template, request

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load IRIS Dataset
iris = sns.load_dataset('iris')

# Split the data into features (X) and labels (y)
X = iris.drop('species', axis=1)
y = iris['species']

# Initialize the Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Train the model on the entire dataset
clf.fit(X, y)

# Prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Create a DataFrame with the new data point
    new_data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })

    # Make a prediction using the trained model
    predicted_species = clf.predict(new_data)

    # Return the predicted species
    return predicted_species[0]

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Use the prediction function to get the predicted species
        predicted_species = predict_species(sepal_length, sepal_width, petal_length, petal_width)

        return render_template('index.html', prediction=predicted_species)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
