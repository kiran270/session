from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

mainteams=['New Zealand','Pakistan','Australia','India','South Africa',
'England','West Indies','Zimbabwe','Bangladesh','Afghanistan','Sri Lanka','Ireland',
'IRE Women','BAN Women','India Women','PAK Women','SRI Women','AUS Women',
'ENG Women','SA Women','NZ Women','Scorchers','Sixers','Hurricanes','Stars','Strikers',
'Renegades','Thunder','Heat']
formats=['WODI','WT20I','MODI','MT20I','MBBL']
app = Flask(__name__)
models={}
def loadmodels(filename, num_overs):
    df = pd.read_csv(filename)
    overs_columns = [f"Over {i}" for i in range(1, num_overs + 1)]
    final_scores = []

    for _, row in df.iterrows():
        final_score = None
        for over in reversed(overs_columns):
            if pd.notna(row[over]) and row[over] != 'N/A':
                over_score = str(row[over]).replace('="', '').replace('"', '')
                if '/' in over_score:
                    try:
                        final_score = int(over_score.split('/')[0])
                    except ValueError:
                        final_score = None
                break
        final_scores.append(final_score)

    df['Final_Score'] = final_scores
    data = []
    for source_row, row in df.dropna(subset=['Final_Score']).iterrows():
        def text_value(column, default=''):
            value = row.get(column, default)
            return default if pd.isna(value) else str(value)

        for over_index, over_column in enumerate(overs_columns):
            over_score = row[over_column]
            if pd.isna(over_score) or over_score == 'N/A':
                continue
            over_score = str(over_score).replace('="', '').replace('"', '')
            if '/' not in over_score:
                continue
            try:
                score, wickets = map(int, over_score.split('/'))
            except ValueError:
                continue
            data.append({
                'Source_Row': int(source_row),
                'Match': text_value('Match', f'Match {source_row + 1}'),
                'Batting_Team': row['Batting_Team'],
                'Bowling_Team': row['Bowling_Team'],
                'Venue': text_value('Venue'),
                'Date': text_value('Date'),
                'Innings': text_value('Innings'),
                'Over': over_index + 1,
                'Score': score,
                'Wickets': wickets,
                'Final_Score': int(row['Final_Score']),
            })

    training_df = pd.DataFrame(data)
    features = training_df[
        ['Batting_Team', 'Bowling_Team', 'Over', 'Score', 'Wickets']
    ]
    X = pd.get_dummies(features, columns=['Batting_Team', 'Bowling_Team'])
    y = training_df['Final_Score']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    model_key = filename.rsplit('.', 1)[0]
    models[model_key] = {
        'model': model,
        'training_df': training_df,
        'X': X,
        'y': y,
    }


def find_similar_matches(
    training_df, batting_team, bowling_team, over, runs, wickets, limit=4
):
    """Return unique historical innings closest to the current match state."""
    candidates = training_df.copy()
    same_batting = candidates['Batting_Team'].eq(batting_team)
    same_bowling = candidates['Bowling_Team'].eq(bowling_team)
    team_penalty = np.select(
        [same_batting & same_bowling, same_batting, same_bowling],
        [0, 8, 10],
        default=22,
    )
    candidates['_distance'] = (
        (candidates['Over'] - over).abs() * 6
        + (candidates['Score'] - runs).abs() * 0.35
        + (candidates['Wickets'] - wickets).abs() * 10
        + team_penalty
    )
    nearest = (
        candidates.sort_values('_distance')
        .drop_duplicates(subset='Source_Row')
        .head(limit)
    )

    return [
        {
            'match': row['Match'],
            'batting_team': row['Batting_Team'],
            'bowling_team': row['Bowling_Team'],
            'venue': row['Venue'],
            'date': row['Date'],
            'innings': row['Innings'],
            'over': int(row['Over']),
            'score': int(row['Score']),
            'wickets': int(row['Wickets']),
            'final_score': int(row['Final_Score']),
            'same_matchup': bool(
                row['Batting_Team'] == batting_team
                and row['Bowling_Team'] == bowling_team
            ),
        }
        for _, row in nearest.iterrows()
    ]

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

    if format_type not in models:
        return jsonify({'error': 'Unsupported match format'}), 400

    model_data = models[format_type]
    X = model_data['X']
    model = model_data['model']

    def build_input(include_batting, include_bowling):
        frame = pd.DataFrame({
            'Over': [over],
            'Score': [runs],
            'Wickets': [wickets],
        })
        for column in X.columns:
            if column.startswith('Batting_Team_'):
                frame[column] = int(
                    include_batting and column == f'Batting_Team_{batting_team}'
                )
            elif column.startswith('Bowling_Team_'):
                frame[column] = int(
                    include_bowling and column == f'Bowling_Team_{bowling_team}'
                )
        return frame.reindex(columns=X.columns, fill_value=0)

    predicted_batting = model.predict(build_input(True, False))[0]
    predicted_bowling = model.predict(build_input(False, True))[0]
    predicted_both = model.predict(build_input(True, True))[0]
    similar_matches = find_similar_matches(
        model_data['training_df'],
        batting_team,
        bowling_team,
        over,
        runs,
        wickets,
    )

    return jsonify({
        'predicted_score_batting': round(predicted_batting, 2),
        'predicted_score_bowling': round(predicted_bowling, 2),
        'predicted_score_both': round(predicted_both, 2),
        'similar_matches': similar_matches,
    })


if __name__ == '__main__':
    loadmodels("MODI.csv",50)
    loadmodels("WODI.csv",50)
    # loadmodels("WODI.csv",50)
    loadmodels("MT20I.csv",20)
    # loadmodels("MBBL.csv",20)
    # loadmodels("WT20I.csv",20)
    # print(models)
    app.run(host='0.0.0.0', port=5000, debug=True)
