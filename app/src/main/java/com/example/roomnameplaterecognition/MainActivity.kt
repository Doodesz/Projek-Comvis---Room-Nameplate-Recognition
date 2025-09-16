package com.example.roomnameplaterecognition // Make sure this matches your project's package name!

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private var module: Module? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)

        // Check for camera permission
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Load the PyTorch model
        loadPyTorchModel()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Set up the Preview use case to display the camera feed
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // Set up the ImageAnalysis use case to process frames
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalyzer())
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(image: ImageProxy) {
            // Convert the ImageProxy to a Bitmap
            val bitmap = imageProxyToBitmap(image)
            if (bitmap != null) {
                // Run inference on the bitmap
                runModelInference(bitmap)
            }
            image.close()
        }
    }

// In MainActivity.kt

    private fun runModelInference(bitmap: Bitmap) {
        module?.let {
            // 👇 THIS IS THE NEW MAGIC LINE! 👇
            // We resize the bitmap from the camera to the 640x640 size our model expects.
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

            // Step 1: Prepare the image for the model (using the RESIZED bitmap!)
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap, // <-- Make sure to use the resizedBitmap here!
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            // Step 2: Run the model!
            val outputTensor = it.forward(IValue.from(inputTensor)).toTensor()
            val results = outputTensor.dataAsFloatArray

            // Step 3: Process the results and update the UI
            // This is where you'll add your logic to find the bounding boxes
            // from the 'results' array and draw them on your OverlayView.

            runOnUiThread {
                // overlayView.setResults(your_parsed_bounding_boxes)
            }
        }
    }

    private fun loadPyTorchModel() {
        try {
            val modelPath = assetFilePath("my_model.ptl") // Change to your model's name!
            module = Module.load(modelPath)
            Log.d(TAG, "PyTorch model loaded successfully!")
        } catch (e: IOException) {
            Log.e(TAG, "Error loading PyTorch model!", e)
        }
    }

    // Helper function to convert ImageProxy to Bitmap
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        if (image.format != ImageFormat.YUV_420_888) {
            Log.e("ImageConverter", "Unsupported image format: ${image.format}")
            return null
        }

        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        //U and V are swapped
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // Rotate the bitmap if necessary
        val matrix = Matrix()
        matrix.postRotate(image.imageInfo.rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    // Helper function to get model file path
    @Throws(IOException::class)
    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath
        assets.open(assetName).use { `is` ->
            FileOutputStream(file).use { os ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (`is`.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
            }
        }
        return file.absolutePath
    }

    // Check if camera permission is granted
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Log.e(TAG, "Permissions not granted by the user.")
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "RoomNameplateRec"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}