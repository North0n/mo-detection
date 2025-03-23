import 'dart:io';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ObjectDetector {
  static const imageSize = 224;
  static const classesCount = 10;
  static const cordsCount = 4;
  static const candidatesCount = 1029;

  late Interpreter _interpreter;
  late List<String> _labels;

  final double confidenceThreshold;
  final double iouThreshold;
  final String datasetName;

  ObjectDetector(this.confidenceThreshold, this.iouThreshold, this.datasetName);

  Future<void> initialize() async {
    _interpreter = await Interpreter.fromAsset(datasetName);

    // Load labels
    _labels = await rootBundle.loadString('assets/models/labels.txt').then((
      content,
    ) {
      return content.split("\n").where((line) => line.isNotEmpty).toList();
    });

    print("Model Loaded Successfully!");
  }

  Uint8List _drawBoundingBoxes(
    img.Image image,
    List<Map<String, dynamic>> detections,
  ) {
    for (var detection in detections) {
      List<double> bbox = detection["bbox"];
      String label = detection["label"];
      double confidence = detection["confidence"];

      // Convert center format (x_center, y_center, width, height) to box corners
      int x1 = ((bbox[0] - bbox[2] / 2) * image.width).toInt();
      int y1 = ((bbox[1] - bbox[3] / 2) * image.height).toInt();
      int x2 = ((bbox[0] + bbox[2] / 2) * image.width).toInt();
      int y2 = ((bbox[1] + bbox[3] / 2) * image.height).toInt();

      // Draw rectangle
      img.drawRect(
        image,
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        color: img.ColorRgb8(255, 0, 0),
        thickness: 3,
      );

      // Draw label text
      String text = "$label ${(confidence * 100).toStringAsFixed(2)}%";
      img.drawString(
        image,
        font: img.arial14,
        x: x1,
        y: y1 - 15,
        text,
        color: img.ColorRgb8(255, 255, 255),
      );
    }

    // Encode image back to Uint8List
    return Uint8List.fromList(img.encodeJpg(image));
  }

  Future<img.Image> getImageBytes(File imageFile) async {
    // Convert image to Float32 List
    Uint8List imageBytes = await imageFile.readAsBytes();
    img.Image? image = img.decodeImage(imageBytes);
    img.Image resizedImage = img.copyResize(
      image!,
      width: imageSize,
      height: imageSize,
    );
    return resizedImage;
  }

  Future<Uint8List> detectedImage(File imageFile) async {
    final image = await getImageBytes(imageFile);
    final objects = await detectObjects(image);
    return _drawBoundingBoxes(image, objects);
  }

  Future<List<Map<String, dynamic>>> detectObjects(img.Image image) async {
    final inputImage = _imageToFloat32List(image);

    // Allocate output buffer
    List<List<List<double>>> output = List.generate(
      1,
      (_) => List.generate(
        classesCount + cordsCount,
        (_) => List.filled(candidatesCount, 0.0),
      ),
    );

    // Run inference
    _interpreter.run([inputImage], output);

    // Process output & apply NMS
    return _processOutput(output);
  }

  List<List<List<double>>> _imageToFloat32List(img.Image image) {
    List<List<List<double>>> buffer = List.generate(
      imageSize,
      (_) => List.generate(imageSize, (_) => List.filled(3, 0.0)),
    );

    for (int y = 0; y < imageSize; y++) {
      for (int x = 0; x < imageSize; x++) {
        final pixel = image.getPixel(x, y);
        buffer[y][x][0] = pixel.getChannel(img.Channel.red) / 255.0;
        buffer[y][x][1] = pixel.getChannel(img.Channel.green) / 255.0;
        buffer[y][x][2] = pixel.getChannel(img.Channel.blue) / 255.0;
      }
    }
    return buffer;
  }

  List<Map<String, dynamic>> _processOutput(List<List<List<double>>> output) {
    List<Map<String, dynamic>> results = [];

    for (int i = 0; i < candidatesCount; i++) {
      List<double> classConfidences =
          output[0]
              .sublist(cordsCount, cordsCount + classesCount)
              .map((e) => e[i])
              .toList();

      // Find the class with the highest confidence
      double maxConfidence = classConfidences.reduce((a, b) => a > b ? a : b);
      int classIndex = classConfidences.indexOf(maxConfidence);

      if (maxConfidence > confidenceThreshold) {
        results.add({
          "label": _labels[classIndex],
          "confidence": maxConfidence,
          "bbox": [
            output[0][0][i], // x_center
            output[0][1][i], // y_center
            output[0][2][i], // width
            output[0][3][i], // height
          ],
        });
      }
    }

    // Apply Non-Maximum Suppression (NMS)
    return _applyNMS(results, iouThreshold, 0.5);
  }

  double _iou(List<double> bbox1, List<double> bbox2) {
    double x1 = max(bbox1[0] - bbox1[2] / 2, bbox2[0] - bbox2[2] / 2);
    double y1 = max(bbox1[1] - bbox1[3] / 2, bbox2[1] - bbox2[3] / 2);
    double x2 = min(bbox1[0] + bbox1[2] / 2, bbox2[0] + bbox2[2] / 2);
    double y2 = min(bbox1[1] + bbox1[3] / 2, bbox2[1] + bbox2[3] / 2);

    double intersection = max(0, x2 - x1) * max(0, y2 - y1);
    double area1 = bbox1[2] * bbox1[3];
    double area2 = bbox2[2] * bbox2[3];
    double unionArea = area1 + area2 - intersection;

    return unionArea > 0 ? intersection / unionArea : 0;
  }

  List<Map<String, dynamic>> _applyNMS(
    List<Map<String, dynamic>> detections,
    double iouThreshold,
    double sigma,
  ) {
    // Sort detections by confidence in descending order
    detections.sort((a, b) => b['confidence'].compareTo(a['confidence']));

    List<Map<String, dynamic>> finalDetections = [];
    while (detections.isNotEmpty) {
      var best = detections.removeAt(0);
      finalDetections.add(best);

      for (var detection in detections) {
        if (best['label'] == detection['label']) {
          double iouValue = _iou(best['bbox'], detection['bbox']);
          if (iouValue > iouThreshold) {
            detection['confidence'] *= exp(-pow(iouValue, 2) / sigma);
          }
        }
      }
      detections.removeWhere((d) => d['confidence'] < 0.01);
    }

    return finalDetections;
  }
}
