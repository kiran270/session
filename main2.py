from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

mainteams=['New Zealand','Pakistan','Australia','India','South Africa',
'England','West Indies','Zimbabwe','Bangladesh','Afghanistan','Sri Lanka','Ireland',
'IRE Women','BAN Women','India Women','PAK Women','SRI Women','AUS Women',
'ENG Women','SA Women','NZ Women']
formats=['WODI','WT20I','MODI','MT20I']
app = Flask(__name__)
models={}
def loadmodels(filename,num_overs):
    file_path = filename
    df = pd.read_csv(file_path)
    overs_columns = [f"Over {i}" for i in range(1,num_overs+1)]
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
    data = []
    for index, row in df_cleaned.iterrows():
        for over_index in range(num_overs):
            over_score = row[overs_columns[over_index]]
            if pd.notna(over_score) and over_score != 'N/A':
                over_score = str(over_score).replace('="', '').replace('"', '')
                if '/' in over_score:
                    try:
                        score, wickets = map(int, over_score.split('/'))
                        data.append({
                            'Batting_Team': row['Batting_Team'],
                            'Bowling_Team': row['Bowling_Team'],
                            'Over': over_index + 1,
                            'Score': score,
                            'Wickets': wickets,
                            'Final_Score': row['Final_Score']
                        })
                    except ValueError:
                        continue

    training_df = pd.DataFrame(data)
    X = training_df[['Batting_Team', 'Bowling_Team', 'Over', 'Score', 'Wickets']]
    X = pd.get_dummies(X, columns=['Batting_Team', 'Bowling_Team'])
    y = training_df['Final_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    temp=filename.split(".")
    models[temp[0]]={"model":model,"training_df":training_df,"X":X,"y":y}
# Route for Home Page
@app.route('/')
def index():
    return render_template("index2.html",mainteams=mainteams,formats=formats)

# Route for Predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    batting_team = data['batting_team'].strip()
    bowling_team = data['bowling_team'].strip()
    format_type = data['format_type'].strip()
    over = int(data['over'])
    runs = int(data['score'])
    wickets = int(data['wickets'])
    # Create input dataframes for each prediction
    input_data_batting = pd.DataFrame({'Over': [over], 'Score': [runs], 'Wickets': [wickets]})
    input_data_bowling = input_data_batting.copy()
    input_data_both = input_data_batting.copy()
    X=models[format_type]["X"]
    model=models[format_type]["model"]
    # One-hot encode batting team
    for col in X.columns:
        if col.startswith('Batting_Team_'):
            input_data_batting[col] = 1 if col == f'Batting_Team_{batting_team}' else 0
        elif col.startswith('Bowling_Team_'):
            input_data_batting[col] = 0  # Not relevant for this prediction

    # One-hot encode bowling team
    for col in X.columns:
        if col.startswith('Bowling_Team_'):
            input_data_bowling[col] = 1 if col == f'Bowling_Team_{bowling_team}' else 0
        elif col.startswith('Batting_Team_'):
            input_data_bowling[col] = 0  # Not relevant for this prediction

    # One-hot encode both batting and bowling teams
    for col in X.columns:
        input_data_both[col] = (
            1 if (col == f'Batting_Team_{batting_team}') and (col == f'Bowling_Team_{bowling_team}') else 0
        )
    # print(input_data_bowling["Bowling_Team_Bangladesh"])
    # Reorder columns to match the training data
    input_data_batting = input_data_batting[X.columns]
    input_data_bowling = input_data_bowling[X.columns]
    input_data_both = input_data_both[X.columns]

    # Make predictions
    predicted_batting = model.predict(input_data_batting)[0]
    predicted_bowling = model.predict(input_data_bowling)[0]
    predicted_both = model.predict(input_data_both)[0]

    return jsonify({
        'predicted_score_batting': round(predicted_batting, 2),
        'predicted_score_bowling': round(predicted_bowling, 2),
        'predicted_score_both': round(predicted_both, 2)
    })

if __name__ == '__main__':
    loadmodels("MODI.csv",50)
    loadmodels("WODI.csv",50)
    # loadmodels("WODI.csv",50)
    loadmodels("MT20I.csv",20)
    # loadmodels("WT20I.csv",20)
    # print(models)
    app.run(host='0.0.0.0', port=5000, debug=True)
