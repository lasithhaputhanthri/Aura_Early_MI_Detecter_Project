import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MI Detector',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: VideoUploaderScreen(),
    );
  }
}

class VideoUploaderScreen extends StatefulWidget {
  const VideoUploaderScreen({super.key});

  @override
  _VideoUploaderScreenState createState() => _VideoUploaderScreenState();
}

class _VideoUploaderScreenState extends State<VideoUploaderScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _videoFile;
  String? _miResult;
  double? _confidence;
  bool _isLoading = false;

    Future<void> _pickVideo(ImageSource source) async {
      final XFile? pickedFile = await _picker.pickVideo(source: source);

      if (pickedFile != null) {
        try {
          // Copy video to a safe location
          final appDir = await getTemporaryDirectory();
          final fileName = path.basename(pickedFile.path);
          final savedFile = await File(pickedFile.path).copy('${appDir.path}/$fileName');

          setState(() {
            _videoFile = savedFile;
            _miResult = null;
            _confidence = null;
          });

          _uploadVideo(savedFile);
        } catch (e) {
          setState(() {
            _miResult = 'Error: $e';
          });
        }
      }
    }

  Future<void> _uploadVideo(File file) async {
    setState(() {
      _isLoading = true;
    });

    final uri = Uri.parse("http://192.168.33.16:5000/upload");  // Flask IP
    final request = http.MultipartRequest("POST", uri);

    final videoBytes = await http.MultipartFile.fromPath(
      'video',
      file.path,
      filename: path.basename(file.path),
    );

    request.files.add(videoBytes);

    try {
      final response = await request.send();

      if (response.statusCode == 200) {
        final respStr = await response.stream.bytesToString();
        final data = json.decode(respStr);

        setState(() {
          _miResult = data['MI'] == true ? "MI" : "Non-MI";
          _confidence = data['confidence'];
        });
      } else {
        setState(() {
          _miResult = "Upload failed";
        });
      }
    } catch (e) {
      setState(() {
        _miResult = "Error: $e";
      });
    }

    setState(() {
      _isLoading = false;
    });
  }

  Widget _buildSquareButton(String label, IconData icon, VoidCallback onPressed) {
    return SizedBox(
      width: 120,
      height: 120,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 36),
            SizedBox(height: 8),
            Text(label, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }

  Widget _buildResultDisplay() {
    if (_isLoading) {
      return CircularProgressIndicator();
    } else if (_miResult != null) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            _miResult!,
            style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold),
          ),
          if (_confidence != null)
            Text(
              "Confidence: ${_confidence!.toStringAsFixed(2)}",
              style: TextStyle(fontSize: 18),
            ),
        ],
      );
    } else {
      return Text(
        "Pick a video to analyze",
        style: TextStyle(fontSize: 18),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Early MI Detector"),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            Expanded(child: Center(child: _buildResultDisplay())),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildSquareButton("Use Camera", Icons.videocam, () => _pickVideo(ImageSource.camera)),
                _buildSquareButton("Choose Gallery", Icons.video_library, () => _pickVideo(ImageSource.gallery)),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
