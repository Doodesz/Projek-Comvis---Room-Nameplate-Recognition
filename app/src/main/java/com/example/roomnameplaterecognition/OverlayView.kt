package com.example.roomnameplaterecognition // Make sure this matches your package name

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.mlkit.vision.objects.DetectedObject
import java.util.LinkedList

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<DetectedObject> = LinkedList<DetectedObject>()
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

        // Now that we receive the correct dimensions, this simple scaling works perfectly.
        val scaleX = width.toFloat() / imageWidth
        val scaleY = height.toFloat() / imageHeight

        for (result in results) {
            val boundingBox = result.boundingBox

            val left = boundingBox.left * scaleX
            val top = boundingBox.top * scaleY
            val right = boundingBox.right * scaleX
            val bottom = boundingBox.bottom * scaleY

            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            if (result.labels.isNotEmpty()) {
                val label = result.labels[0].text
                val score = String.format("%.2f", result.labels[0].confidence)
                val drawableText = "$label $score"
                canvas.drawText(drawableText, left, top - 10f, textPaint)
            }
        }
    }

    fun setResults(
        detectionResults: List<DetectedObject>,
        imageHeight: Int,
        imageWidth: Int,
    ) {
        results = detectionResults
        this.imageHeight = imageHeight
        this.imageWidth = imageWidth
        invalidate()
    }
}