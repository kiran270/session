import 'dart:convert';
import 'package:http/http.dart' as http;

import '../models/prediction.dart';

/// Talks to the Flask backend defined in `main2.py`.
class PredictionService {
  /// Base URL of the running Flask server.
  ///
  /// - Android emulator reaches the host machine at 10.0.2.2
  /// - iOS simulator / desktop / web can use localhost
  /// Change this to your machine's LAN IP when testing on a real device.
  final String baseUrl;

  PredictionService({this.baseUrl = 'http://52.5.107.23'});

  Future<Prediction> predict({
    required String battingTeam,
    required String bowlingTeam,
    required String formatKey,
    required int over,
    required int score,
    required int wickets,
  }) async {
    final uri = Uri.parse('$baseUrl/predict');
    final response = await http.post(
      uri,
      body: {
        'batting_team': battingTeam,
        'bowling_team': bowlingTeam,
        'format_type': formatKey,
        'over': over.toString(),
        'score': score.toString(),
        'wickets': wickets.toString(),
      },
    ).timeout(const Duration(seconds: 20));

    if (response.statusCode != 200) {
      throw Exception('Server error (${response.statusCode}).');
    }
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return Prediction.fromJson(data);
  }
}
