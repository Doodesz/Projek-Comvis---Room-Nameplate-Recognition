package com.example.roomnameplaterecognition // Make sure package name is correct

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private val boxPaint = Paint()
    private val textPaint = Paint()

    init {
        boxPaint.color = Color.parseColor("#FF6F61") // A nice coral color
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 8f

        textPaint.color = Color.WHITE
        textPaint.textSize = 40f
        textPaint.style = Paint.Style.FILL
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate() // This tells the view to redraw itself
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val scaleX = width.toFloat() / 640f
        val scaleY = height.toFloat() / 640f

        for (box in results) {
            val scaledRect = RectF(
                box.x1 * scaleX,
                box.y1 * scaleY,
                box.x2 * scaleX,
                box.y2 * scaleY
            )
            canvas.drawRect(scaledRect, boxPaint)

            val text = "${box.clsName} (${"%.2f".format(box.cnf)})"
            canvas.drawText(text, scaledRect.left, scaledRect.top - 10, textPaint)
        }
    }
}