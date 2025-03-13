import 'dart:io' show File;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'object_detector.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.orange),
      ),
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ObjectDetector detector = ObjectDetector(0.25, 0.45);
  File? _image;
  Uint8List? _processedImage;

  @override
  void initState() {
    super.initState();
    detector.initialize().then((_) {
      setState(() {});
    });
  }

  Future<void> _pickImage() async {
    final pickedFile = await ImagePicker().pickImage(
      source: ImageSource.gallery,
    );
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _processedImage = null; // Reset processed image
      });
      _runDetection();
    }
  }

  Future<void> _captureImage() async {
    final pickedFile = await ImagePicker().pickImage(
      source: ImageSource.camera,
    );
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _processedImage = null; // Reset processed image
      });
      _runDetection();
    }
  }

  Future<void> _runDetection() async {
    if (_image == null) return;
    final processed = await detector.detectedImage(_image!);
    setState(() {
      _processedImage = processed;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Object Detection')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image != null
                ? Image.file(
                  _image!,
                  width: ObjectDetector.imageSize.toDouble(),
                  height: ObjectDetector.imageSize.toDouble(),
                  fit: BoxFit.cover,
                )
                : SizedBox(
                  width: ObjectDetector.imageSize.toDouble(),
                  height: ObjectDetector.imageSize.toDouble(),
                  child: Center(child: Text("No image selected")),
                ),
            SizedBox(height: 10),
            _processedImage != null
                ? Image.memory(_processedImage!)
                : Text("Pick an image to detect objects"),
            SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: _pickImage,
                  child: Text("Pick Image"),
                ),
                SizedBox(width: 10),
                ElevatedButton(
                  onPressed: _captureImage,
                  child: Text("Take Photo"),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
