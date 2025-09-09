package com.example.roomnameplaterecognition // Make sure this matches your package name!

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.roomnameplaterecognition.databinding.ActivityMainBinding
import com.google.common.util.concurrent.ListenableFuture
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private lateinit var cameraExecutor: ExecutorService
    private var interpreter: Interpreter? = null

    // Constants for the model
    private val MODEL_INPUT_WIDTH = 640
    private val MODEL_INPUT_HEIGHT = 640
    private val CONFIDENCE_THRESHOLD = 0.5f
    private val IOU_THRESHOLD = 0.5f // For Non-Max Suppression

    private lateinit var labels: List<String>


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        requestCameraPermission()
        loadModelAndLabels()
    }

    private fun loadModelAndLabels() {
        try {
            val modelBuffer = FileUtil.loadMappedFile(this, "model.tflite")
            interpreter = Interpreter(modelBuffer, Interpreter.Options())
            labels = FileUtil.loadLabels(this, "labels.txt")
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading model or labels: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e("MainActivity", "Error loading model or labels", e)
        }
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                startCamera()
            } else {
                Toast.makeText(this, "Permission denied! Can't use the camera.", Toast.LENGTH_SHORT).show()
            }
        }

    private fun requestCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }


    private fun startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(binding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalysis.Analyzer { imageProxy ->
                        if (interpreter == null) {
                            imageProxy.close()
                            return@Analyzer
                        }

                        val bitmap = imageProxy.toBitmap()
                        val imageProcessor = ImageProcessor.Builder()
                            .add(ResizeOp(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, ResizeOp.ResizeMethod.BILINEAR))
                            .add(CastOp(DataType.FLOAT32))
                            .build()
                        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
                        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 25200, 85), DataType.FLOAT32)

                        interpreter?.run(tensorImage.buffer, outputBuffer.buffer.rewind())
                        val results = processOutput(outputBuffer)

                        runOnUiThread {
                            binding.overlay.setResults(
                                results,
                                imageProxy.height,
                                imageProxy.width
                            )
                        }
                        imageProxy.close()
                    })
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e("MainActivity", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processOutput(buffer: TensorBuffer): List<BoundingBox> {
        val outputArray = buffer.floatArray
        val boundingBoxes = mutableListOf<BoundingBox>()
        val numElementsPerDetection = 85
        val numDetections = outputArray.size / numElementsPerDetection

        for (i in 0 until numDetections) {
            val offset = i * numElementsPerDetection
            val confidence = outputArray[offset + 4]

            // Let's log the top confidence scores to see what the model thinks it sees!
            if (confidence > 0.25f) { // Lower threshold just for logging
                Log.d("MainActivity", "Detected something with confidence: $confidence")
            }

            if (confidence >= CONFIDENCE_THRESHOLD) {
                var maxClassScore = 0f
                var cls = -1
                for (j in 0 until 80) {
                    val score = outputArray[offset + 5 + j]
                    if (score > maxClassScore) {
                        maxClassScore = score
                        cls = j
                    }
                }

                if (maxClassScore > CONFIDENCE_THRESHOLD) {
                    // THE FIX IS HERE: We are removing the multiplication!
                    val cx = outputArray[offset]
                    val cy = outputArray[offset + 1]
                    val w = outputArray[offset + 2]
                    val h = outputArray[offset + 3]

                    val x1 = cx - w / 2
                    val y1 = cy - h / 2
                    val x2 = cx + w / 2
                    val y2 = cy + h / 2

                    if (cls != -1) {
                        boundingBoxes.add(
                            BoundingBox(x1, y1, x2, y2, cx, cy, w, h, confidence * maxClassScore, cls, labels[cls])
                        )
                    }
                }
            }
        }
        return nonMaxSuppression(boundingBoxes)
    }

    // ✨ THIS IS OUR NEW HELPER FUNCTION FOR NMS ✨
    private fun nonMaxSuppression(boxes: List<BoundingBox>): List<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }
        val selectedBoxes = mutableListOf<BoundingBox>()

        for (box in sortedBoxes) {
            var shouldSelect = true
            for (selectedBox in selectedBoxes) {
                if (calculateIoU(box, selectedBox) > IOU_THRESHOLD) {
                    shouldSelect = false
                    break
                }
            }
            if (shouldSelect) {
                selectedBoxes.add(box)
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = max(box1.x1, box2.x1)
        val y1 = max(box1.y1, box2.y1)
        val x2 = min(box1.x2, box2.x2)
        val y2 = min(box1.y2, box2.y2)

        val intersectionArea = max(0f, x2 - x1) * max(0f, y2 - y1)
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        val unionArea = box1Area + box2Area - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }


    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        interpreter?.close()
    }
}