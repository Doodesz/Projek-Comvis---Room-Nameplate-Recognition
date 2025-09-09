package com.example.roomnameplaterecognition // Make sure this matches your package name!

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

// This is our new custom data class to hold detection results
class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val cnf: Float,
    val cls: Int,
    val clsName: String
)

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<BoundingBox> = listOf()
    private var boxPaint = Paint()
    private var textPaint = Paint()

    private var scaleFactor: Float = 1f

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

        for (box in results) {
            val left = box.x1 * scaleFactor
            val top = box.y1 * scaleFactor
            val right = box.x2 * scaleFactor
            val bottom = box.y2 * scaleFactor

            canvas.drawRect(left, top, right, bottom, boxPaint)
            val drawableText = "${box.clsName} ${String.format("%.2f", box.cnf)}"
            canvas.drawText(drawableText, left, top - 10f, textPaint)
        }
    }

    fun setResults(
        boundingBoxes: List<BoundingBox>,
        imageHeight: Int,
        imageWidth: Int,
    ) {
        results = boundingBoxes
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
        invalidate()
    }
}