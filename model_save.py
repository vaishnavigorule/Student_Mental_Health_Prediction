import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("student_mental_health_survey.csv")

# Create encoders
pressure_encoder = LabelEncoder()
social_encoder = LabelEncoder()
activity_encoder = LabelEncoder()
diet_encoder = LabelEncoder()
target_encoder = LabelEncoder()

# Apply encoders
df['Academic Pressure'] = pressure_encoder.fit_transform(df['Academic Pressure'])
df['Social Life'] = social_encoder.fit_transform(df['Social Life'])
df['Physical Activity'] = activity_encoder.fit_transform(df['Physical Activity'])
df['Diet'] = diet_encoder.fit_transform(df['Diet'])
df['Mental Health Issues'] = target_encoder.fit_transform(df['Mental Health Issues'])

# Features & target
X = df[['Academic Pressure', 'Social Life', 'Physical Activity', 'Diet']]
y = df['Mental Health Issues']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'student_mental_health_model.pkl')
joblib.dump(pressure_encoder, 'pressure_encoder.pkl')
joblib.dump(social_encoder, 'social_encoder.pkl')
joblib.dump(activity_encoder, 'activity_encoder.pkl')
joblib.dump(diet_encoder, 'diet_encoder.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

print("✅ Model and encoders saved successfully.")
