import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import zipfile
import matplotlib.pyplot as plt

zip_path = r"heart+disease.zip"
extract_path = r"heart+disease"


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


dataset_path = r"heart+disease\cleveland.data"
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]


data = pd.read_csv(dataset_path, header=None, names=columns, na_values='?')
print(data.head())
print(data.isnull().sum())
data = data.dropna()


X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


print("\nPlease answer the following questions:")
try:
    user_data = {
        "age": int(input("Enter your age: ")),
        "sex": int(input("Enter your sex (1=Male, 0=Female): ")),
        "cp": int(input("Enter chest pain type (0-3): ")),
        "trestbps": int(input("Enter resting blood pressure: ")),
        "chol": int(input("Enter cholesterol level: ")),
        "fbs": int(input("Fasting blood sugar > 120 mg/dl (1=True, 0=False): ")),
        "restecg": int(input("Enter resting ECG results (0-2): ")),
        "thalach": int(input("Enter maximum heart rate achieved: ")),
        "exang": int(input("Exercise induced angina (1=Yes, 0=No): ")),
        "oldpeak": float(input("Enter ST depression induced by exercise: ")),
        "slope": int(input("Enter slope of the peak exercise ST segment (0-2): ")),
        "ca": int(input("Number of major vessels (0-3): ")),
        "thal": int(input("Enter thalassemia (1-3): ")),
    }

    user_df = pd.DataFrame([user_data], columns=X.columns)

    # Get probability of heart disease 
    probability = model.predict_proba(user_df)[:, 1][0]

    # Convert probability to percentage
    percentage = probability * 100

    #Visualization of the risk percentage
    plt.figure()
    plt.ylabel('Risk Percentage')
    plt.title('User Heart Disease Risk Percentage')
    plt.ylim([0,100]) # Set y-axis limit to 0-100 for percentage
    # plt.bar(["User Risk"], [percentage], color='blue')
    
    #met plot graph
    if percentage >= 70:
        risk_level = "High Risk"
        bar_color = 'red'
    elif percentage >= 40:
        risk_level = "Moderate Risk"
        bar_color = 'orange'
    else:
        risk_level = "Low Risk"
        bar_color = 'green'
    plt.bar(["Risk"], [percentage], color=bar_color) #Color change according to risk level

    plt.text(0, -10, risk_level, 
    ha='center', va='top', 
    fontsize=15, fontweight='bold',
    color=bar_color
    ) #Risk level headline below the bar

    plt.tight_layout() #To prevent labels from being cut off
    print(f"\nYour estimated heart disease risk is: {percentage:.2f}%")
    plt.show()

    

except ValueError as e:
    print("\nInvalid input. Please enter numbers only.")
except IndexError as e:
    print("Please provide all the inputs")

except ValueError as e:
    print("\nInvalid input. Please enter numbers only.")
except IndexError as e:
    print("Please provide all the inputs")



    user_data_values = list(user_data.values())
    feature_names = list(user_data.keys())
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, user_data_values, color='orange')
    plt.ylabel('Value')
    plt.title('User Input Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

except ValueError as e:
    print("\nInvalid input. Please enter numbers only.")

pickle.dump(model,open("predictivemodel.pkl","wb"))

str.find()