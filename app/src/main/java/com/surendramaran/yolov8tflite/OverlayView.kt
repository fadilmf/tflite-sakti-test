package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList
import kotlin.math.max

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var bounds = Rect()

    private var sourceImageWidth: Float = 1f
    private var sourceImageHeight: Float = 1f

    init {
        initPaints()
    }

    fun clear() {
        results = listOf()
        sourceImageWidth = 1f // Reset juga
        sourceImageHeight = 1f // Reset juga
        // textPaint.reset() // Sebaiknya initPaints dipanggil ulang saja
        // textBackgroundPaint.reset()
        // boxPaint.reset()
        invalidate()
        initPaints() // Panggil ulang untuk memastikan paint terinisialisasi dengan benar
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // Hindari pembagian dengan nol atau penskalaan salah jika dimensi belum di-set
        if (sourceImageWidth <= 1f || sourceImageHeight <= 1f || results.isEmpty()) {
            return
        }

        // Faktor skala dari koordinat gambar asli ke koordinat OverlayView
        val scaleX = width.toFloat() / sourceImageWidth     // width adalah OverlayView.width
        val scaleY = height.toFloat() / sourceImageHeight   // height adalah OverlayView.height

        results.forEach { box ->
            // box.x1, box.y1, dll. adalah koordinat piksel di gambar ASLI

            // Skalakan koordinat ke ruang OverlayView
            val displayLeft = box.x1 * scaleX
            val displayTop = box.y1 * scaleY
            val displayRight = box.x2 * scaleX
            val displayBottom = box.y2 * scaleY

            // Gambar persegi panjang dengan koordinat yang sudah diskalakan
            // Canvas akan secara otomatis meng-clip bagian yang di luar batasnya
            canvas.drawRect(displayLeft, displayTop, displayRight, displayBottom, boxPaint)

            val drawableText = box.clsName // Pastikan BoundingBox punya clsName
            // Jika menggunakan properti `cls` (int) dan `labels` list dari Detector,
            // Anda mungkin perlu cara untuk mendapatkan nama kelas di sini.
            // Untuk sementara, asumsikan `box.clsName` sudah benar.

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textRectHeight = bounds.height() // Tinggi actual dari teks dirender

            // Tentukan posisi teks (misalnya di pojok kiri atas bounding box)
            // Pastikan teks juga menggunakan koordinat yang sudah diskalakan
            val textDrawX = displayLeft
            val textDrawY = displayTop // Ini akan jadi batas atas background teks

            // Background untuk teks
            canvas.drawRect(
                textDrawX,
                textDrawY,
                textDrawX + textWidth + BOUNDING_RECT_TEXT_PADDING,
                textDrawY + textRectHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Teks di atas background
            // textDrawY + textRectHeight adalah baseline untuk drawText jika textDrawY adalah y-atas
            canvas.drawText(drawableText, textDrawX + (BOUNDING_RECT_TEXT_PADDING / 2f), textDrawY + textRectHeight + (BOUNDING_RECT_TEXT_PADDING / 2f) - bounds.bottom, textPaint)
            // Penyesuaian bounds.bottom mungkin diperlukan untuk alignment vertikal yang lebih baik dalam background
            // Atau lebih sederhana:
            // canvas.drawText(drawableText, textDrawX, textDrawY + textRectHeight, textPaint)
            // Ini menempatkan baseline teks di bagian bawah area background. Sesuaikan padding jika perlu.
        }
    }

    // Modifikasi setResults untuk menerima dimensi gambar asli
    fun setResults(boundingBoxes: List<BoundingBox>, originalImgWidth: Int, originalImgHeight: Int) {
        results = boundingBoxes
        this.sourceImageWidth = originalImgWidth.toFloat()
        this.sourceImageHeight = originalImgHeight.toFloat()
        invalidate() // Minta OverlayView untuk menggambar ulang
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
