/// A historical innings with a match state close to the current input.
class ComparableMatch {
  final String match;
  final String battingTeam;
  final String bowlingTeam;
  final String venue;
  final String date;
  final String innings;
  final int over;
  final int score;
  final int wickets;
  final int finalScore;
  final bool sameMatchup;

  const ComparableMatch({
    required this.match,
    required this.battingTeam,
    required this.bowlingTeam,
    required this.venue,
    required this.date,
    required this.innings,
    required this.over,
    required this.score,
    required this.wickets,
    required this.finalScore,
    required this.sameMatchup,
  });

  factory ComparableMatch.fromJson(Map<String, dynamic> json) {
    int number(String key) => (json[key] as num?)?.toInt() ?? 0;
    String text(String key) => json[key]?.toString() ?? '';

    return ComparableMatch(
      match: text('match'),
      battingTeam: text('batting_team'),
      bowlingTeam: text('bowling_team'),
      venue: text('venue'),
      date: text('date'),
      innings: text('innings'),
      over: number('over'),
      score: number('score'),
      wickets: number('wickets'),
      finalScore: number('final_score'),
      sameMatchup: json['same_matchup'] == true,
    );
  }
}

/// Result returned by the `/predict` endpoint.
class Prediction {
  final double battingScore;
  final double bowlingScore;
  final double bothScore;
  final List<ComparableMatch> similarMatches;

  const Prediction({
    required this.battingScore,
    required this.bowlingScore,
    required this.bothScore,
    this.similarMatches = const [],
  });

  factory Prediction.fromJson(Map<String, dynamic> json) {
    double parse(dynamic value) => (value as num).toDouble();
    final matches = json['similar_matches'] as List<dynamic>? ?? const [];

    return Prediction(
      battingScore: parse(json['predicted_score_batting']),
      bowlingScore: parse(json['predicted_score_bowling']),
      bothScore: parse(json['predicted_score_both']),
      similarMatches: matches
          .map((item) => ComparableMatch.fromJson(
                Map<String, dynamic>.from(item as Map),
              ))
          .toList(),
    );
  }
}
