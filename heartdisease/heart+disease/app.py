from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('predictivemodel.pkl', 'rb'))

# Load and preprocess the data (do this outside the function)
dataset_path = r"heart+disease\processed.cleveland.data"
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
data = pd.read_csv(dataset_path, header=None, names=columns, na_values='?')
data = data.dropna()
X = data.drop('target', axis=1) 

data['high_cholesterol'] = (data['chol'] > 200).astype(int) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            high_cholesterol: int(request.form['high_cholesterol']) # type: ignore
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

            user_df = pd.DataFrame([user_data], columns=X.columns)

            # Get probability of heart disease 
            probability = model.predict_proba(user_df)[:, 1][0]
            percentage = probability * 100

            # Determine risk level
            if percentage >= 70:
                risk_level = "High Risk"
                bar_color = 'red'
                template_name = 'high_risk.html' 
            elif percentage >= 40:
                risk_level = "Moderate Risk"
                bar_color = 'orange'
                template_name = 'moderate_risk.html' 
            else:
                risk_level = "Safe Zone"
                bar_color = 'green'
                template_name = 'low_risk.html' 

            # Create the plot
            plt.figure()
            plt.ylabel('Risk Percentage')
            plt.title('User Heart Disease Risk Percentage')
            plt.ylim([0, 100])
            plt.bar(["Risk"], [percentage], color=bar_color)

            plt.text(0, -10, risk_level, 
                    ha='center', va='top', 
                    fontsize=15, fontweight='bold',
                    color=bar_color)

            plt.tight_layout()

            # Save the plot to a buffer
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')

            return render_template(template_name, prediction_text=f"Your estimated heart disease risk is: {percentage:.2f}% ({risk_level})", plot_url=plot_url)

        except ValueError as e:
            return render_template('error.html', error_message="Invalid input. Please enter numbers only.")
        except IndexError as e:
            return render_template('error.html', error_message="Please provide all the inputs.")

if __name__ == '__main__':
    app.run(debug=True)