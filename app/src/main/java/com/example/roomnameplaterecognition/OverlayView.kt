package com.example.roomnameplaterecognition // Make sure this matches your package name

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.task.vision.detector.Detection
import java.util.LinkedList

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<Detection> = LinkedList<Detection>()
    private var boxPaint = Paint()
    private var textPaint = Paint()

    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    init {
        initPaints()
    }

    private fun initPaints() {
        boxPaint.color = Color.GREEN
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 8f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f
        textPaint.color = Color.GREEN
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        if (results.isEmpty()) {
            return
        }

        // This is the crucial rotation correction logic.
        // It checks if the image is wider than it is tall (landscape) while the
        // view is taller than it is wide (portrait), and vice-versa.
        val isImageRotated = imageWidth > imageHeight && width < height || imageWidth < imageHeight && width > height

        val scale: Float
        val offsetX: Float
        val offsetY: Float

        if (isImageRotated) {
            // If the image is rotated, we swap the dimensions for scaling calculations.
            scale = width.toFloat() / imageHeight
            offsetX = 0f
            offsetY = (height - imageWidth * scale) / 2
        } else {
            scale = width.toFloat() / imageWidth
            offsetX = 0f
            offsetY = (height - imageHeight * scale) / 2
        }


        for (result in results) {
            val boundingBox = result.boundingBox
            var left: Float
            var top: Float
            var right: Float
            var bottom: Float

            if (isImageRotated) {
                // If rotated, we transform the coordinates from the landscape image
                // to the portrait view.
                left = boundingBox.top * scale + offsetX
                top = (imageWidth - boundingBox.right) * scale + offsetY
                right = boundingBox.bottom * scale + offsetX
                bottom = (imageWidth - boundingBox.left) * scale + offsetY
            } else {
                // If not rotated, we just apply the scale and offset.
                left = boundingBox.left * scale + offsetX
                top = boundingBox.top * scale + offsetY
                right = boundingBox.right * scale + offsetX
                bottom = boundingBox.bottom * scale + offsetY
            }

            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            if (result.categories.isNotEmpty()) {
                val label = result.categories[0].label
                val score = String.format("%.2f", result.categories[0].score)
                val drawableText = "$label $score"
                canvas.drawText(drawableText, left, top - 10f, textPaint)
            }
        }
    }

    fun setResults(
        detectionResults: List<Detection>,
        imageHeight: Int,
        imageWidth: Int,
    ) {
        results = detectionResults
        this.imageHeight = imageHeight
        this.imageWidth = imageWidth
        invalidate()
    }
}