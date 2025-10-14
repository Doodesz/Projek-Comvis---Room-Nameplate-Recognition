package com.example.roomnameplaterecognition // Make sure this is your package name

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStreamReader
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.tasks.await
import kotlin.coroutines.resumeWithException

class MainActivity : AppCompatActivity() {

    // UI Views
    private lateinit var cameraContainer: ConstraintLayout
    private lateinit var resultContainer: ConstraintLayout
    private lateinit var previewView: PreviewView
    private lateinit var resultImageView: ImageView
    private lateinit var captureButton: Button
    private lateinit var backButton: Button

    private lateinit var cameraExecutor: ExecutorService // CameraX and Model variables

    private lateinit var textRecognizer: TextRecognizer // OCR

    private lateinit var ocrTextView: TextView

    private var module: Module? = null
    private var imageCapture: ImageCapture? = null
    private val classNames = mutableListOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Find all our UI elements
        cameraContainer = findViewById(R.id.camera_container)
        resultContainer = findViewById(R.id.result_container)
        previewView = findViewById(R.id.previewView)
        resultImageView = findViewById(R.id.resultImageView)
        captureButton = findViewById(R.id.captureButton)
        backButton = findViewById(R.id.backButton)
        ocrTextView = findViewById(R.id.ocrTextView)

        // Set up button listeners
        captureButton.setOnClickListener { takePhoto() }
        backButton.setOnClickListener { showCameraView() }

        // OCR
        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

        // Standard setup
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
        cameraExecutor = Executors.newSingleThreadExecutor()
        loadPyTorchModel()
        loadClassNames()
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val image = takePictureAsynchronously(imageCapture)
                val capturedBitmap = imageProxyToBitmap(image)
                image.close()

                if (capturedBitmap != null) {
                    val boxes = runModelInference(capturedBitmap)
                    var finalOcrText = "" // Variabel untuk menyimpan teks akhir

                    if (boxes.isNotEmpty()) {
                        // Kita hanya proses kotak pertama yang terdeteksi untuk kesederhanaan
                        val mainBox = boxes[0]

                        // 👇 LOGIKA BARU DIMULAI DI SINI 👇
                        when (mainBox.clsName) {
                            "toilet", "toilet difabel", "toilet laki-laki", "toilet perempuan" -> {
                                finalOcrText = "(Toilet)"
                            }
                            "tempat wudhu" -> {
                                finalOcrText = "(Tempat Wudhu)"
                            }
                            else -> {
                                // Jika bukan kelas spesial, baru jalankan OCR
                                val boxesWithText = runOcrOnBoundingBoxes(capturedBitmap, listOf(mainBox))
                                finalOcrText = boxesWithText[0].recognizedText.ifBlank { "Teks tidak terbaca" }
                            }
                        }
                    } else {
                        finalOcrText = "Objek tidak ditemukan"
                    }

                    val resultBitmap = drawBoxesOnBitmap(capturedBitmap, boxes)

                    withContext(Dispatchers.Main) {
                        // Update TextView dengan teks final
                        ocrTextView.text = finalOcrText
                        showResultView(resultBitmap)
                    }
                }
            } catch (exc: Exception) {
                Log.e(TAG, "Photo capture atau processing gagal", exc)
            }
        }
    }

    private suspend fun takePictureAsynchronously(imageCapture: ImageCapture): ImageProxy {
        return suspendCancellableCoroutine { continuation ->
            imageCapture.takePicture(
                cameraExecutor, // You can also use ContextCompat.getMainExecutor(this)
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        Log.d(TAG, "Photo capture succeeded!")
                        continuation.resume(image) {
                            // This block is called if the coroutine is cancelled.
                            // We should close the image to prevent memory leaks.
                            image.close()
                        }
                    }

                    override fun onError(exc: ImageCaptureException) {
                        Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                        continuation.resumeWithException(exc)
                    }
                }
            )
        }
    }

    private suspend fun runOcrOnBoundingBoxes(originalBitmap: Bitmap, boxes: List<BoundingBox>): List<BoundingBox> {
        // This calculates how to scale the box coordinates from the small 640x640 image
        // back to the big, original high-quality image.
        val scaleX = originalBitmap.width / 640f
        val scaleY = originalBitmap.height / 640f

        // We loop through every box that the YOLO model found.
        for (box in boxes) {
            try {
                // We calculate the exact coordinates to crop from the big image.
                val left = (box.x1 * scaleX).toInt().coerceAtLeast(0)
                val top = (box.y1 * scaleY).toInt().coerceAtLeast(0)
                val width = ((box.x2 - box.x1) * scaleX).toInt().coerceAtMost(originalBitmap.width - left)
                val height = ((box.y2 - box.y1) * scaleY).toInt().coerceAtMost(originalBitmap.height - top)

                // We create a small bitmap of just the detected area (the nameplate).
                val croppedBitmap = Bitmap.createBitmap(originalBitmap, left, top, width, height)

                // We prepare this small bitmap for ML Kit.
                val image = InputImage.fromBitmap(croppedBitmap, 0)

                // We run the OCR and wait for the result.
                val result = textRecognizer.process(image).await()

                // We save the text we found into our BoundingBox object.
                // We also replace any line breaks with a space to keep it on one line.
                box.recognizedText = result.text.replace("\n", " ")

                Log.d(TAG, "OCR Result for box ${box.clsName}: ${box.recognizedText}")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to crop or run OCR on a box", e)
                box.recognizedText = "OCR Error" // Mark that OCR failed for this box
            }
        }
        return boxes // Return the list of boxes, now updated with text!
    }

    // This function now returns the list of detected boxes
    private fun runModelInference(bitmap: Bitmap): List<BoundingBox> {
        return module?.let {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
            val noMean = floatArrayOf(0.0f, 0.0f, 0.0f)
            val noStd = floatArrayOf(1.0f, 1.0f, 1.0f)
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, noMean, noStd)

            val outputTensor = it.forward(IValue.from(inputTensor)).toTensor()
            val results = outputTensor.dataAsFloatArray

            parseYoloOutput(results)
        } ?: emptyList() // Return an empty list if the model is null
    }

    private fun drawBoxesOnBitmap(bitmap: Bitmap, boxes: List<BoundingBox>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val boxPaint = Paint().apply {
            color = Color.parseColor("#FF6F61")
            style = Paint.Style.STROKE
            strokeWidth = 8f
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 50f
            style = Paint.Style.FILL
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        val scaleX = mutableBitmap.width.toFloat() / 640f
        val scaleY = mutableBitmap.height.toFloat() / 640f

        for (box in boxes) {
            val scaledRect = RectF(
                box.x1 * scaleX, box.y1 * scaleY,
                box.x2 * scaleX, box.y2 * scaleY
            )
            canvas.drawRect(scaledRect, boxPaint)

            // KEMBALI HANYA MENAMPILKAN NAMA KELAS
            val text = "${box.clsName} (${"%.2f".format(box.cnf)})"
            canvas.drawText(text, scaledRect.left, scaledRect.top - 20, textPaint)
        }
        return mutableBitmap
    }

    private fun showResultView(resultBitmap: Bitmap) {
        resultImageView.setImageBitmap(resultBitmap)
        cameraContainer.visibility = View.GONE
        resultContainer.visibility = View.VISIBLE
    }

    private fun showCameraView() {
        cameraContainer.visibility = View.VISIBLE
        resultContainer.visibility = View.GONE
    }

    // --- All other functions below this line are the same, no changes needed ---
    // (startCamera, loadPyTorchModel, parseYoloOutput, etc.)
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            imageCapture = ImageCapture.Builder().build()
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
            } catch (exc: Exception) { Log.e(TAG, "Use case binding failed", exc) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun loadClassNames() {
        try {
            val reader = BufferedReader(InputStreamReader(assets.open("labels.txt")))
            var line: String?
            while (reader.readLine().also { line = it } != null) { classNames.add(line!!) }
            reader.close()
        } catch (e: Exception) { Log.e("Labels", "Error loading class names!", e) }
    }

    private fun loadPyTorchModel() {
        try {
            val modelPath = assetFilePath("my_model.ptl")
            module = Module.load(modelPath)
            Log.d(TAG, "PyTorch model loaded successfully!")
        } catch (e: IOException) { Log.e(TAG, "Error loading PyTorch model!", e) }
    }

    private fun parseYoloOutput(results: FloatArray): List<BoundingBox> {
        val numBoxes = 8400
        val numClasses = 9
        val boxes = mutableListOf<BoundingBox>()
        for (i in 0 until numBoxes) {
            var maxScore = 0f
            var maxIndex = -1
            for (j in 0 until numClasses) {
                val score = results[(j + 4) * numBoxes + i]
                if (score > maxScore) {
                    maxScore = score
                    maxIndex = j
                }
            }
            if (maxScore > 0.5f) {
                val cx = results[0 * numBoxes + i]
                val cy = results[1 * numBoxes + i]
                val w = results[2 * numBoxes + i]
                val h = results[3 * numBoxes + i]
                val x1 = cx - w / 2; val y1 = cy - h / 2; val x2 = cx + w / 2; val y2 = cy + h / 2
                boxes.add(BoundingBox(x1, y1, x2, y2, cx, cy, w, h, maxScore, maxIndex, if (maxIndex in classNames.indices) classNames[maxIndex] else "Unknown"))
            }
        }
        return applyNMS(boxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>): List<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }
        val selectedBoxes = mutableListOf<BoundingBox>()
        val active = BooleanArray(boxes.size) { true }
        var numActive = active.size
        for (i in sortedBoxes.indices) {
            if (active[i]) {
                val boxA = sortedBoxes[i]; selectedBoxes.add(boxA)
                if (numActive == 1) break
                for (j in i + 1 until sortedBoxes.size) {
                    if (active[j]) {
                        val boxB = sortedBoxes[j]
                        if (calculateIoU(boxA.getRect(), boxB.getRect()) > 0.4f) {
                            active[j] = false; numActive--
                        }
                    }
                }
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(rectA: RectF, rectB: RectF): Float {
        val xA = max(rectA.left, rectB.left); val yA = max(rectA.top, rectB.top)
        val xB = min(rectA.right, rectB.right); val yB = min(rectA.bottom, rectB.bottom)
        val interArea = max(0f, xB - xA) * max(0f, yB - yA)
        val boxAArea = (rectA.right - rectA.left) * (rectA.bottom - rectA.top)
        val boxBArea = (rectB.right - rectB.left) * (rectB.bottom - rectB.top)
        return interArea / (boxAArea + boxBArea - interArea)
    }

    // In MainActivity.kt

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        val bitmap: Bitmap?
        if (image.format == ImageFormat.JPEG) {
            // If the image is already a JPEG, it's super easy to convert
            val buffer = image.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } else if (image.format == ImageFormat.YUV_420_888) {
            // If it's the YUV format, we use our robust converter
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
            val imageBytes = out.toByteArray()
            bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        } else {
            // If the format is something else, we can't handle it.
            Log.e("ImageConverter", "Unsupported image format: ${image.format}")
            return null
        }

        // Finally, we rotate the bitmap to the correct orientation
        val matrix = Matrix()
        matrix.postRotate(image.imageInfo.rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    @Throws(IOException::class)
    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath
        assets.open(assetName).use { `is` -> FileOutputStream(file).use { os ->
            val buffer = ByteArray(4 * 1024); var read: Int
            while (`is`.read(buffer).also { read = it } != -1) { os.write(buffer, 0, read) }
            os.flush()
        }
        }
        return file.absolutePath
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) { startCamera() } else { finish() }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        textRecognizer.close()
    }

    companion object {
        private const val TAG = "RoomNameplateRec"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}