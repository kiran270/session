from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load Data and Train Model
file_path = 'WODI.csv'
df = pd.read_csv(file_path)

# Data Preprocessing
overs_columns = [f"Over {i}" for i in range(1, 51)]
final_scores = []

for index, row in df.iterrows():
    final_score = None
    for over in reversed(overs_columns):
        if pd.notna(row[over]) and row[over] != 'N/A':
            over_score = str(row[over]).replace('="', '').replace('"', '')
            if '/' in over_score:
                score, _ = over_score.split('/')
                try:
                    final_score = int(score)
                except ValueError:
                    final_score = None
            break
    final_scores.append(final_score)

df['Final_Score'] = final_scores
df_cleaned = df.dropna(subset=['Final_Score'])

# Prepare Training Data
data = []
for index, row in df_cleaned.iterrows():
    for over_index in range(50):
        over_score = row[overs_columns[over_index]]
        if pd.notna(over_score) and over_score != 'N/A':
            over_score = str(over_score).replace('="', '').replace('"', '')
            if '/' in over_score:
                try:
                    score, wickets = map(int, over_score.split('/'))
                    data.append({
                        'Team': row['Team'],
                        'Over': over_index + 1,
                        'Score': score,
                        'Wickets': wickets,
                        'Final_Score': row['Final_Score']
                    })
                except ValueError:
                    continue

training_df = pd.DataFrame(data)
X = training_df[['Team', 'Over', 'Score', 'Wickets']]
X = pd.get_dummies(X, columns=['Team'])
y = training_df['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route for Predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    team_name = data['team'].strip()
    over = int(data['over'])
    score = int(data['score'])
    wickets = int(data['wickets'])

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Over': [over],
        'Score': [score],
        'Wickets': [wickets]
    })

    if team_name:  # If team name is provided
        for team_col in [col for col in X.columns if col.startswith('Team_')]:
            input_data[team_col] = 1 if team_col == f'Team_{team_name}' else 0
    else:  # If team name is empty, use the full dataset to predict
        # For this case, we could either return multiple predictions or handle it differently
        predicted_scores = model.predict(X)  # Predict for all teams in the dataset
        return jsonify({'predicted_scores': predicted_scores.tolist()})

    # Ensure all columns from training data are in input_data
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the training data
    input_data = input_data[X.columns]

    # Make prediction for the provided team input
    predicted_score = model.predict(input_data)[0]

    return jsonify({'predicted_score': round(predicted_score, 2)})
if __name__ == '__main__':
    app.run(debug=True)
