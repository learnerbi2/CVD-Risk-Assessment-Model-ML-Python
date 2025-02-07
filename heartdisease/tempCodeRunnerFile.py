from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('predictivemodel.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        user_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        user_df = pd.DataFrame([user_data])

        # Get probability of heart disease
        probability = model.predict_proba(user_df)[:, 1][0]
        risk_percentage = probability * 100

        # Determine risk level
        if risk_percentage >= 70:
            risk_level = "High Risk"
        elif risk_percentage >= 40:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        return render_template('index.html', prediction_text=f"Your estimated heart disease risk is: {risk_percentage:.2f}% ({risk_level})")

if __name__ == '__main__':
    app.run(debug=True)