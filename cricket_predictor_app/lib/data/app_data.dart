/// Static reference data mirrored from the Flask backend (`main2.py`).
class AppData {
  /// Teams the backend was trained on.
  static const List<String> teams = [
    'New Zealand', 'Pakistan', 'Australia', 'India', 'South Africa',
    'England', 'West Indies', 'Zimbabwe', 'Bangladesh', 'Afghanistan',
    'Sri Lanka', 'Ireland',
    'IRE Women', 'BAN Women', 'India Women', 'PAK Women', 'SRI Women',
    'AUS Women', 'ENG Women', 'SA Women', 'NZ Women',
    'Scorchers', 'Sixers', 'Hurricanes', 'Stars', 'Strikers',
    'Renegades', 'Thunder', 'Heat',
  ];

  /// Formats keyed to the model name the backend expects and total overs.
  static const List<MatchFormat> formats = [
    MatchFormat(label: 'Men ODI', modelKey: 'MODI', overs: 50),
    MatchFormat(label: 'Women ODI', modelKey: 'WODI', overs: 50),
    MatchFormat(label: 'Men T20I', modelKey: 'MT20I', overs: 20),
  ];
}

class MatchFormat {
  final String label;
  final String modelKey;
  final int overs;

  const MatchFormat({
    required this.label,
    required this.modelKey,
    required this.overs,
  });
}
