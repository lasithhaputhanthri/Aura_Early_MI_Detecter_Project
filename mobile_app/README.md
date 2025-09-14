# Mobile App (Flutter)

This is the Flutter mobile client for the Early MI Detector Project.

---

## 🚀 Setup Instructions

### 1. Install Flutter
Download and install Flutter from the official docs:  
[https://docs.flutter.dev/get-started/install](https://docs.flutter.dev/get-started/install)

Verify installation:
```bash
flutter doctor
```

### 2. Navigate to the mobile app folder
```bash
cd mobile_app
```

### 3. Install dependencies
```bash
flutter pub get
```

### 4. Run the app

- **Android (with USB debugging enabled):**
  ```bash
  flutter run
  ```

- **Web (Chrome):**
  ```bash
  flutter run -d chrome
  ```

- **iOS (Mac required):**
  ```bash
  flutter run
  ```

---

## ⚡ Notes

- Ensure the backend Flask server is running before using the app:
  ```bash
  python flask_server/run.py
  ```

- Update the API base URL in the Flutter project if needed (check `lib/` folder for config).

- Do not commit large build files. The `.gitignore` already excludes:
  ```
  mobile_app/build/
  mobile_app/.dart_tool/
  mobile_app/.packages
  mobile_app/pubspec.lock
  ```
