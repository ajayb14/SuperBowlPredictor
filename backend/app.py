from flask import Flask, jsonify, send_from_directory
import pandas as pd
import joblib

app = Flask(__name__, static_folder="../frontend", static_url_path="/")

# Load the trained model
model = joblib.load("../model/rf_nfl_superbowl_winner_prediction.pkl")

# Load the new team stats
team_stats = pd.read_csv("../data/final_team_stats.csv")

# Filter to 2024 season data
latest = team_stats[team_stats['season'] == 2024].copy()

features = ['pass_yards', 'rush_yards', 'total_yards', 'points_scored', 'points_allowed', 'point_differential', 'sb_wins_last_5']

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["GET"])
def predict():
    if latest.empty:
        return jsonify({"error": "No data available for the 2024 season."}), 400

    X_latest = latest[features]
    # Probability of SB win
    probabilities = model.predict_proba(X_latest)[:, 1]
    latest['probability'] = probabilities

    # Get top predicted team
    top_team = latest.loc[latest['probability'].idxmax()]
    return jsonify({
        "team": top_team['team'],
        "season": int(top_team["season"]) + 1, # predicting for next season
        "probability": top_team['probability']
    })
if __name__ == "__main__":
    app.run(debug=True)
