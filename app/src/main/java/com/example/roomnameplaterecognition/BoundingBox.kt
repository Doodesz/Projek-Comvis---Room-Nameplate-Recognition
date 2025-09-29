// In BoundingBox.kt
package com.example.roomnameplaterecognition // Make sure package name is correct

import android.graphics.RectF

data class BoundingBox(
    // Object Detection
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
    val clsName: String,

    // OCR
    var recognizedText: String = ""
) {
    fun getRect(): RectF = RectF(x1, y1, x2, y2)
}