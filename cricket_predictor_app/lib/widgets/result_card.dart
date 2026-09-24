import 'package:flutter/material.dart';
import '../models/prediction.dart';
import '../theme/app_theme.dart';
import 'glass_card.dart';

/// Scoreboard-style summary of the three model signals.
class ResultCard extends StatelessWidget {
  final Prediction prediction;

  const ResultCard({super.key, required this.prediction});

  @override
  Widget build(BuildContext context) {
    final scores = [
      prediction.battingScore,
      prediction.bowlingScore,
      prediction.bothScore,
    ]..sort();

    return GlassCard(
      padding: const EdgeInsets.all(14),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(
                  children: [
                    Icon(Icons.auto_awesome, color: AppColors.primary, size: 13),
                    SizedBox(width: 5),
                    Text(
                      'AI PROJECTION',
                      style: TextStyle(
                        color: AppColors.primary,
                        fontSize: 9,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1,
                      ),
                    ),
                  ],
                ),
              ),
              const Spacer(),
              const Icon(Icons.bolt_rounded, color: AppColors.accent, size: 16),
              const Text(
                ' LIVE MODEL',
                style: TextStyle(
                  color: AppColors.textMuted,
                  fontSize: 9,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.8,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'PROJECTED TOTAL',
                      style: TextStyle(
                        color: AppColors.textMuted,
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1.2,
                      ),
                    ),
                    TweenAnimationBuilder<double>(
                      tween: Tween(begin: 0, end: prediction.bothScore),
                      duration: const Duration(milliseconds: 750),
                      curve: Curves.easeOutCubic,
                      builder: (_, value, __) => Row(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          Text(
                            value.toStringAsFixed(0),
                            style: const TextStyle(
                              color: AppColors.textDark,
                              fontSize: 48,
                              height: 1.05,
                              fontWeight: FontWeight.w900,
                              letterSpacing: -2,
                            ),
                          ),
                          const Padding(
                            padding: EdgeInsets.only(bottom: 6, left: 5),
                            child: Text(
                              'RUNS',
                              style: TextStyle(
                                color: AppColors.primary,
                                fontSize: 10,
                                fontWeight: FontWeight.w800,
                                letterSpacing: 1,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 10),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFF192B36), Color(0xFF152133)],
                  ),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: AppColors.accent.withOpacity(0.25)),
                ),
                child: Column(
                  children: [
                    const Text(
                      'MODEL RANGE',
                      style: TextStyle(
                        color: AppColors.textMuted,
                        fontSize: 8,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.8,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      '${scores.first.round()}–${scores.last.round()}',
                      style: const TextStyle(
                        color: AppColors.accent,
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const Divider(height: 18, color: Color(0xFF27364D)),
          Row(
            children: [
              Expanded(
                child: _ModelSignal(
                  label: 'BATTING SIGNAL',
                  value: prediction.battingScore,
                  icon: Icons.sports_cricket_rounded,
                  color: AppColors.primary,
                ),
              ),
              Container(width: 1, height: 34, color: const Color(0xFF27364D)),
              Expanded(
                child: _ModelSignal(
                  label: 'BOWLING SIGNAL',
                  value: prediction.bowlingScore,
                  icon: Icons.track_changes_rounded,
                  color: AppColors.accent,
                ),
              ),
            ],
          ),
          if (prediction.similarMatches.isNotEmpty) ...[
            const Divider(height: 18, color: Color(0xFF27364D)),
            Material(
              color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(12),
                onTap: () => _showComparableMatches(context),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 5),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.history_rounded,
                        color: AppColors.primary,
                        size: 18,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          '${prediction.similarMatches.length} CLOSEST HISTORICAL INNINGS',
                          style: const TextStyle(
                            color: AppColors.textDark,
                            fontSize: 9,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.8,
                          ),
                        ),
                      ),
                      const Text(
                        'VIEW',
                        style: TextStyle(
                          color: AppColors.accent,
                          fontSize: 9,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      const SizedBox(width: 3),
                      const Icon(
                        Icons.chevron_right_rounded,
                        color: AppColors.accent,
                        size: 18,
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  void _showComparableMatches(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: AppColors.surface,
      isScrollControlled: true,
      showDragHandle: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(26)),
      ),
      builder: (_) => _ComparableMatchesSheet(
        matches: prediction.similarMatches,
      ),
    );
  }
}

class _ComparableMatchesSheet extends StatefulWidget {
  final List<ComparableMatch> matches;

  const _ComparableMatchesSheet({required this.matches});

  @override
  State<_ComparableMatchesSheet> createState() =>
      _ComparableMatchesSheetState();
}

class _ComparableMatchesSheetState extends State<_ComparableMatchesSheet> {
  String _team = '';
  String _ground = '';

  List<String> get _teams => widget.matches
      .expand((match) => [match.battingTeam, match.bowlingTeam])
      .where((team) => team.isNotEmpty)
      .toSet()
      .toList()
    ..sort();

  List<String> get _grounds => widget.matches
      .map((match) => match.venue)
      .where((ground) => ground.isNotEmpty)
      .toSet()
      .toList()
    ..sort();

  List<ComparableMatch> get _filtered => widget.matches.where((match) {
        final teamMatches = _team.isEmpty ||
            match.battingTeam == _team ||
            match.bowlingTeam == _team;
        final groundMatches = _ground.isEmpty || match.venue == _ground;
        return teamMatches && groundMatches;
      }).toList();

  @override
  Widget build(BuildContext context) {
    final matches = _filtered;
    final filterActive = _team.isNotEmpty || _ground.isNotEmpty;

    return SafeArea(
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxHeight: MediaQuery.sizeOf(context).height * 0.78,
        ),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(18, 0, 18, 18),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Expanded(
                    child: Text(
                      'Historical match explorer',
                      style: TextStyle(
                        color: AppColors.textDark,
                        fontSize: 18,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ),
                  if (filterActive)
                    TextButton.icon(
                      onPressed: () => setState(() {
                        _team = '';
                        _ground = '';
                      }),
                      icon: const Icon(Icons.restart_alt_rounded, size: 16),
                      label: const Text('RESET'),
                    ),
                ],
              ),
              const SizedBox(height: 3),
              Text(
                '${matches.length} of ${widget.matches.length} comparable innings',
                style: const TextStyle(
                  color: AppColors.textMuted,
                  fontSize: 11,
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: _FilterDropdown(
                      label: 'TEAM',
                      icon: Icons.groups_2_rounded,
                      value: _team,
                      allLabel: 'All teams',
                      options: _teams,
                      onChanged: (value) => setState(() => _team = value),
                    ),
                  ),
                  const SizedBox(width: 9),
                  Expanded(
                    child: _FilterDropdown(
                      label: 'GROUND',
                      icon: Icons.stadium_rounded,
                      value: _ground,
                      allLabel:
                          _grounds.isEmpty ? 'No ground data' : 'All grounds',
                      options: _grounds,
                      enabled: _grounds.isNotEmpty,
                      onChanged: (value) => setState(() => _ground = value),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 13),
              Flexible(
                child: matches.isEmpty
                    ? const Center(
                        child: Padding(
                          padding: EdgeInsets.symmetric(vertical: 34),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                Icons.filter_alt_off_rounded,
                                color: AppColors.textMuted,
                                size: 30,
                              ),
                              SizedBox(height: 8),
                              Text(
                                'No innings match these filters',
                                style: TextStyle(color: AppColors.textMuted),
                              ),
                            ],
                          ),
                        ),
                      )
                    : ListView.separated(
                        shrinkWrap: true,
                        itemCount: matches.length,
                        separatorBuilder: (_, __) =>
                            const SizedBox(height: 9),
                        itemBuilder: (_, index) =>
                            _ComparableMatchTile(match: matches[index]),
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _FilterDropdown extends StatelessWidget {
  final String label;
  final IconData icon;
  final String value;
  final String allLabel;
  final List<String> options;
  final ValueChanged<String> onChanged;
  final bool enabled;

  const _FilterDropdown({
    required this.label,
    required this.icon,
    required this.value,
    required this.allLabel,
    required this.options,
    required this.onChanged,
    this.enabled = true,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            color: AppColors.textMuted,
            fontSize: 9,
            fontWeight: FontWeight.w800,
            letterSpacing: 1,
          ),
        ),
        const SizedBox(height: 5),
        DropdownButtonFormField<String>(
          value: value,
          isExpanded: true,
          dropdownColor: AppColors.surfaceHigh,
          icon: const Icon(Icons.expand_more_rounded, size: 18),
          style: const TextStyle(
            color: AppColors.textDark,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
          decoration: InputDecoration(
            enabled: enabled,
            prefixIconConstraints: const BoxConstraints(minWidth: 36),
            prefixIcon: Icon(
              icon,
              color: enabled ? AppColors.primary : AppColors.textMuted,
              size: 16,
            ),
          ),
          items: [
            DropdownMenuItem(value: '', child: Text(allLabel)),
            ...options.map(
              (option) => DropdownMenuItem(
                value: option,
                child: Text(
                  option,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ),
          ],
          onChanged: enabled
              ? (selected) => onChanged(selected ?? '')
              : null,
        ),
      ],
    );
  }
}

class _ComparableMatchTile extends StatelessWidget {
  final ComparableMatch match;

  const _ComparableMatchTile({required this.match});

  @override
  Widget build(BuildContext context) {
    final metadata = [match.date, match.venue]
        .where((value) => value.isNotEmpty)
        .join(' • ');

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surfaceHigh,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: match.sameMatchup
              ? AppColors.primary.withOpacity(0.35)
              : const Color(0xFF27364D),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: AppColors.background,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              '${match.finalScore}',
              style: const TextStyle(
                color: AppColors.primary,
                fontSize: 16,
                fontWeight: FontWeight.w900,
              ),
            ),
          ),
          const SizedBox(width: 11),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${match.battingTeam} vs ${match.bowlingTeam}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: AppColors.textDark,
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  'Over ${match.over}: ${match.score}/${match.wickets}  →  Final ${match.finalScore}',
                  style: const TextStyle(
                    color: AppColors.accent,
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                if (metadata.isNotEmpty) ...[
                  const SizedBox(height: 2),
                  Text(
                    metadata,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      color: AppColors.textMuted,
                      fontSize: 9,
                    ),
                  ),
                ],
              ],
            ),
          ),
          if (match.sameMatchup)
            const Icon(Icons.verified_rounded, color: AppColors.primary, size: 17),
        ],
      ),
    );
  }
}

class _ModelSignal extends StatelessWidget {
  final String label;
  final double value;
  final IconData icon;
  final Color color;

  const _ModelSignal({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(icon, color: color, size: 18),
        const SizedBox(width: 8),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: const TextStyle(
                color: AppColors.textMuted,
                fontSize: 8,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.7,
              ),
            ),
            Text(
              value.toStringAsFixed(0),
              style: const TextStyle(
                color: AppColors.textDark,
                fontSize: 18,
                fontWeight: FontWeight.w800,
              ),
            ),
          ],
        ),
      ],
    );
  }
}
