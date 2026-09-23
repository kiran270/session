# Cricket Score Predictor (Flutter)

A beautiful Flutter mobile app that predicts a cricket team's final score. It connects to the existing Flask ML backend (`main2.py`) in the parent folder.

## Features

- Clean, modern Material 3 UI with a cricket pitch-green theme
- Gradient header and soft card-based layout
- Format picker (Men ODI, Women ODI, Men T20I)
- Team dropdowns, +/- steppers for over & wickets, score input
- Live prediction results (batting model, bowling model, and combined)

## Screens & structure

```
lib/
├── main.dart                 # App entry + MaterialApp
├── theme/app_theme.dart      # Colors, gradients, ThemeData
├── data/app_data.dart        # Teams & formats (mirrors the backend)
├── models/prediction.dart    # Response model
├── services/prediction_service.dart  # HTTP client for /predict
├── screens/home_screen.dart  # Main prediction screen
└── widgets/                  # Reusable UI (cards, dropdown, stepper, result)
```

## Running it

1. Start the Flask backend from the parent folder:
   ```bash
   python main2.py
   ```
   It listens on `0.0.0.0:5000`.

2. Get Flutter packages:
   ```bash
   cd cricket_predictor_app
   flutter pub get
   ```

3. Point the app at your backend. Open `lib/services/prediction_service.dart`
   and set `baseUrl`:
   - **Android emulator:** `http://10.0.2.2:5000` (default)
   - **iOS simulator / desktop:** `http://localhost:5000`
   - **Real device:** `http://<your-computer-LAN-IP>:5000`

4. Run the app:
   ```bash
   flutter run
   ```

## Notes

- The `format_type` sent to the backend uses the model keys `MODI`, `WODI`, `MT20I`, matching the models loaded in `main2.py`.
- To predict on Android against localhost, `10.0.2.2` is the emulator's alias for your host machine.
