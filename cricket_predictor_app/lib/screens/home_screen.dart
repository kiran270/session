import 'package:flutter/material.dart';

import '../data/app_data.dart';
import '../models/prediction.dart';
import '../services/prediction_service.dart';
import '../theme/app_theme.dart';
import '../widgets/glass_card.dart';
import '../widgets/labeled_dropdown.dart';
import '../widgets/number_stepper.dart';
import '../widgets/result_card.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _service = PredictionService();

  MatchFormat _format = AppData.formats.first;
  String? _battingTeam = AppData.teams.first;
  String? _bowlingTeam = AppData.teams[1];
  int _over = 1;
  int _score = 0;
  int _wickets = 0;
  bool _loading = false;
  String? _error;
  Prediction? _result;

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    if (_battingTeam == _bowlingTeam) {
      setState(() => _error = 'Choose two different teams.');
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });

    try {
      final prediction = await _service.predict(
        battingTeam: _battingTeam!,
        bowlingTeam: _bowlingTeam!,
        formatKey: _format.modelKey,
        over: _over,
        score: _score,
        wickets: _wickets,
      );
      if (mounted) setState(() => _result = prediction);
    } catch (_) {
      if (mounted) {
        setState(() => _error = 'Prediction server is unavailable. Try again.');
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      appBar: _buildAppBar(),
      body: Stack(
        children: [
          const Positioned.fill(
            child: DecoratedBox(
              decoration: BoxDecoration(gradient: AppColors.stadiumGradient),
            ),
          ),
          Positioned(
            top: -90,
            right: -80,
            child: _glow(AppColors.accent, 230),
          ),
          Positioned(
            bottom: -120,
            left: -100,
            child: _glow(AppColors.primary, 280),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
              child: Column(
                children: [
                  _buildMatchLab(),
                  const SizedBox(height: 10),
                  _buildPredictButton(),
                  const SizedBox(height: 10),
                  Expanded(
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 350),
                      switchInCurve: Curves.easeOutCubic,
                      child: _error != null
                          ? Align(
                              key: const ValueKey('error'),
                              alignment: Alignment.topCenter,
                              child: _buildError(),
                            )
                          : _result != null
                              ? Align(
                                  key: ValueKey(_result!.bothScore),
                                  alignment: Alignment.topCenter,
                                  child: ResultCard(prediction: _result!),
                                )
                              : Align(
                                  key: const ValueKey('idle'),
                                  alignment: Alignment.topCenter,
                                  child: _buildIdleState(),
                                ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  PreferredSizeWidget _buildAppBar() {
    return AppBar(
      toolbarHeight: 60,
      elevation: 0,
      scrolledUnderElevation: 0,
      automaticallyImplyLeading: false,
      backgroundColor: AppColors.background,
      titleSpacing: 14,
      title: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              gradient: AppColors.primaryGradient,
              borderRadius: BorderRadius.circular(12),
              boxShadow: [
                BoxShadow(
                  color: AppColors.primary.withOpacity(0.22),
                  blurRadius: 12,
                ),
              ],
            ),
            child: const Icon(
              Icons.sports_cricket_rounded,
              color: AppColors.background,
              size: 22,
            ),
          ),
          const SizedBox(width: 10),
          const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'PITCHPULSE',
                style: TextStyle(
                  color: AppColors.textDark,
                  fontSize: 17,
                  fontWeight: FontWeight.w900,
                  letterSpacing: 0.8,
                ),
              ),
              Text(
                'CRICKET INTELLIGENCE',
                style: TextStyle(
                  color: AppColors.textMuted,
                  fontSize: 8,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.3,
                ),
              ),
            ],
          ),
        ],
      ),
      actions: [
        Container(
          margin: const EdgeInsets.only(right: 14),
          padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
          decoration: BoxDecoration(
            color: AppColors.primary.withOpacity(0.1),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: AppColors.primary.withOpacity(0.22)),
          ),
          child: const Row(
            children: [
              CircleAvatar(radius: 3, backgroundColor: AppColors.primary),
              SizedBox(width: 6),
              Text(
                'AI LIVE',
                style: TextStyle(
                  color: AppColors.primary,
                  fontSize: 9,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.7,
                ),
              ),
            ],
          ),
        ),
      ],
      bottom: const PreferredSize(
        preferredSize: Size.fromHeight(1),
        child: Divider(height: 1, color: Color(0xFF1C293B)),
      ),
    );
  }

  Widget _buildMatchLab() {
    return GlassCard(
      padding: const EdgeInsets.all(14),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              const Text(
                'MATCH LAB',
                style: TextStyle(
                  color: AppColors.textDark,
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 1.1,
                ),
              ),
              const Spacer(),
              Text(
                '${_format.overs} OVER FORMAT',
                style: const TextStyle(
                  color: AppColors.textMuted,
                  fontSize: 9,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          _buildFormatSelector(),
          const SizedBox(height: 11),
          Row(
            children: [
              Expanded(
                child: LabeledDropdown<String>(
                  label: 'Batting',
                  icon: Icons.sports_cricket_rounded,
                  value: _battingTeam,
                  items: AppData.teams,
                  itemLabel: (team) => team,
                  onChanged: (team) => setState(() => _battingTeam = team),
                ),
              ),
              Container(
                width: 30,
                height: 30,
                margin: const EdgeInsets.fromLTRB(7, 18, 7, 0),
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: AppColors.background,
                  shape: BoxShape.circle,
                  border: Border.all(color: const Color(0xFF304058)),
                ),
                child: const Text(
                  'VS',
                  style: TextStyle(
                    color: AppColors.accent,
                    fontSize: 9,
                    fontWeight: FontWeight.w900,
                  ),
                ),
              ),
              Expanded(
                child: LabeledDropdown<String>(
                  label: 'Bowling',
                  icon: Icons.track_changes_rounded,
                  value: _bowlingTeam,
                  items: AppData.teams,
                  itemLabel: (team) => team,
                  onChanged: (team) => setState(() => _bowlingTeam = team),
                ),
              ),
            ],
          ),
          const SizedBox(height: 11),
          Row(
            children: [
              Expanded(
                child: NumberStepper(
                  label: 'Over',
                  icon: Icons.timelapse_rounded,
                  value: _over,
                  min: 1,
                  max: _format.overs,
                  onChanged: (value) => setState(() => _over = value),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(child: _buildScoreField()),
              const SizedBox(width: 8),
              Expanded(
                child: NumberStepper(
                  label: 'Wickets',
                  icon: Icons.close_rounded,
                  value: _wickets,
                  min: 0,
                  max: 10,
                  onChanged: (value) => setState(() => _wickets = value),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildFormatSelector() {
    return Container(
      height: 39,
      padding: const EdgeInsets.all(3),
      decoration: BoxDecoration(
        color: AppColors.background,
        borderRadius: BorderRadius.circular(13),
        border: Border.all(color: const Color(0xFF223149)),
      ),
      child: Row(
        children: AppData.formats.map((format) {
          final selected = format == _format;
          return Expanded(
            child: Semantics(
              button: true,
              selected: selected,
              label: format.label,
              child: GestureDetector(
                onTap: () => setState(() {
                  _format = format;
                  if (_over > format.overs) _over = format.overs;
                }),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 180),
                  alignment: Alignment.center,
                  decoration: BoxDecoration(
                    gradient: selected ? AppColors.primaryGradient : null,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    format.label,
                    style: TextStyle(
                      color: selected ? AppColors.background : AppColors.textMuted,
                      fontSize: 10,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildScoreField() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'SCORE',
          style: TextStyle(
            color: AppColors.textMuted,
            fontSize: 10,
            fontWeight: FontWeight.w700,
            letterSpacing: 1,
          ),
        ),
        const SizedBox(height: 5),
        SizedBox(
          height: 46,
          child: TextFormField(
            initialValue: _score.toString(),
            keyboardType: TextInputType.number,
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: AppColors.textDark,
              fontSize: 17,
              fontWeight: FontWeight.w800,
            ),
            decoration: const InputDecoration(contentPadding: EdgeInsets.zero),
            onChanged: (value) => _score = int.tryParse(value) ?? 0,
          ),
        ),
      ],
    );
  }

  Widget _buildPredictButton() {
    return Semantics(
      button: true,
      label: 'Generate score prediction',
      child: Opacity(
        opacity: _loading ? 0.65 : 1,
        child: Container(
          height: 50,
          decoration: BoxDecoration(
            gradient: AppColors.primaryGradient,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: AppColors.primary.withOpacity(0.2),
                blurRadius: 18,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          child: Material(
            color: Colors.transparent,
            child: InkWell(
              borderRadius: BorderRadius.circular(16),
              onTap: _loading ? null : _predict,
              child: Center(
                child: _loading
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2.4,
                          color: AppColors.background,
                        ),
                      )
                    : const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.auto_awesome_rounded,
                            color: AppColors.background,
                            size: 19,
                          ),
                          SizedBox(width: 9),
                          Text(
                            'GENERATE PREDICTION',
                            style: TextStyle(
                              color: AppColors.background,
                              fontSize: 12,
                              fontWeight: FontWeight.w900,
                              letterSpacing: 1,
                            ),
                          ),
                        ],
                      ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildIdleState() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
      decoration: BoxDecoration(
        color: AppColors.surface.withOpacity(0.55),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFF223149)),
      ),
      child: const Row(
        children: [
          Icon(Icons.radar_rounded, color: AppColors.accent, size: 21),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              'Set the live match state and let the model read the game.',
              style: TextStyle(
                color: AppColors.textMuted,
                fontSize: 11,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildError() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: const Color(0xFF321A24),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF633047)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline_rounded, color: Color(0xFFFF6B91), size: 20),
          const SizedBox(width: 9),
          Expanded(
            child: Text(
              _error!,
              style: const TextStyle(color: Color(0xFFFFA0B8), fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }

  Widget _glow(Color color, double size) {
    return IgnorePointer(
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          gradient: RadialGradient(
            colors: [color.withOpacity(0.11), color.withOpacity(0)],
          ),
        ),
      ),
    );
  }
}
