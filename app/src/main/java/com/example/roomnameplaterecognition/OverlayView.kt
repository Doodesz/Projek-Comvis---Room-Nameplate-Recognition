package com.example.roomnameplaterecognition // Make sure this matches your package name

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import org.tensorflow.lite.task.vision.detector.Detection
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<Detection> = listOf()
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

        // This is the new, smarter scaling logic!
        // It calculates how to fit the image onto the screen while maintaining its shape.
        val scaleFactor = min(width.toFloat() / imageWidth, height.toFloat() / imageHeight)

        // It then calculates the empty space on the sides or top/bottom.
        val scaledImageWidth = imageWidth * scaleFactor
        val scaledImageHeight = imageHeight * scaleFactor
        val offsetX = (width - scaledImageWidth) / 2
        val offsetY = (height - scaledImageHeight) / 2

        for (result in results) {
            val boundingBox = result.boundingBox

            // It uses the scale factor and offsets to draw the box in the perfect spot.
            val left = boundingBox.left * scaleFactor + offsetX
            val top = boundingBox.top * scaleFactor + offsetY
            val right = boundingBox.right * scaleFactor + offsetX
            val bottom = boundingBox.bottom * scaleFactor + offsetY

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
        invalidate() // This tells the view to redraw itself
    }
}