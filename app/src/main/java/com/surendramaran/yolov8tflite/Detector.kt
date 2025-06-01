package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener
) {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val INPUT_STANDARD_DEVIATION = 255f

    companion object {
        private const val TAG = "Detector"
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(
            boundingBoxes: List<BoundingBox>,
            inferenceTime: Long,
            originalImageWidth: Int,
            originalImageHeight: Int
        )
    }

    init {
        try {
            setupInterpreter(useGpu = false)
            setupLabels()
            Log.d(TAG, "Detector initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize detector", e)
            throw e
        }
    }

    private fun setupInterpreter(useGpu: Boolean = false) {
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }

            if (useGpu) {
                val compatibilityList = CompatibilityList()
                if (compatibilityList.isDelegateSupportedOnThisDevice) {
                    try {
                        gpuDelegate = GpuDelegate()
                        options.addDelegate(gpuDelegate!!)
                        Log.d(TAG, "GPU delegate added successfully")
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to create GPU delegate, falling back to CPU", e)
                        gpuDelegate?.close()
                        gpuDelegate = null
                    }
                } else {
                    Log.w(TAG, "GPU delegate not supported on this device")
                }
            }

            val model = loadModelFile()
            interpreter = Interpreter(model, options)

            // Dapatkan dimensi input
            val inputShape = interpreter!!.getInputTensor(0).shape()
            tensorHeight = inputShape[1]
            tensorWidth = inputShape[2]
            numChannel = if (inputShape.size > 3) inputShape[3] else 3
            numElements = tensorHeight * tensorWidth * numChannel

            Log.d(TAG, "Model input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "Tensor dims: ${tensorWidth}x${tensorHeight}x${numChannel}")

            // Debug semua output tensor yang tersedia
            val outputCount = interpreter!!.outputTensorCount
            Log.d(TAG, "Total outputTensorCount = $outputCount")
            for (i in 0 until outputCount) {
                val outShape = interpreter!!.getOutputTensor(i).shape()
                val outName  = interpreter!!.getOutputTensor(i).name()
                Log.d(TAG, "Output tensor idx $i, name=$outName, shape=${outShape.contentToString()}")
            }
            // Asumsikan raw output ada di index 0: [1, 5, 22869]
            // Jika ternyata ada lebih dari satu, silakan sesuaikan indexnya.

        } catch (e: Exception) {
            Log.e(TAG, "Error setting up interpreter", e)
            cleanup()
            throw e
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        return try {
            FileUtil.loadMappedFile(context, modelPath)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model file: $modelPath", e)
            throw e
        }
    }

    private fun setupLabels() {
        try {
            labels.clear()
            val inputStream = context.assets.open(labelPath)
            inputStream.bufferedReader().useLines { lines ->
                lines.forEach { line ->
                    val trimmed = line.trim()
                    if (trimmed.isNotEmpty()) {
                        labels.add(trimmed)
                    }
                }
            }
            Log.d(TAG, "Loaded ${labels.size} labels")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels from: $labelPath", e)
            throw e
        }
    }

    fun detect(bitmap: Bitmap) {
        if (interpreter == null) {
            Log.w(TAG, "Interpreter not initialized")
            detectorListener.onEmptyDetect()
            return
        }

        try {
            val startTime = System.currentTimeMillis()

            // Resize + prepare input
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, tensorWidth, tensorHeight, false)
            val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

            // Output buffer: diasumsikan raw output di index 0, shape [1, 5, 22869]
            val outShape = interpreter!!.getOutputTensor(0).shape()
            // outShape = [1, 5, numPredictions]
            val numPredictions = outShape[2]
            val rawOutput = Array(1) { Array(outShape[1]) { FloatArray(numPredictions) } }

            // Run inference
            interpreter!!.run(inputBuffer, rawOutput)

            val inferenceTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "Raw output dims: ${rawOutput.size} x ${rawOutput[0].size} x ${rawOutput[0][0].size}")

            // Proses output raw menjadi bounding boxes
            val boundingBoxes = processRawOutput(rawOutput[0], bitmap.width, bitmap.height)

            if (boundingBoxes.isEmpty()) {
                detectorListener.onEmptyDetect()
            } else {
                detectorListener.onDetect(boundingBoxes, inferenceTime, bitmap.width, bitmap.height)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error during detection", e)
            detectorListener.onEmptyDetect()
        }
    }

    /**
     * rawOut: Array<FloatArray> dengan dimensi [5][numPredictions],
     * di mana index 0 = x_center_norm,
     *       index 1 = y_center_norm,
     *       index 2 = width_norm,
     *       index 3 = height_norm,
     *       index 4 = object_confidence (skalar per prediksi).
     *
     * originalWidth/Height: resolusi asli bitmap
     */
    private fun processRawOutput(
        rawOut: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int
    ): List<BoundingBox> {
        val candidates = mutableListOf<BoundingBox>()
        val numPreds = rawOut[0].size

        try {
            Log.d(TAG, "Processing raw output with $numPreds predictions")
            for (i in 0 until numPreds) {
                val xNorm = rawOut[0][i]
                val yNorm = rawOut[1][i]
                val wNorm = rawOut[2][i]
                val hNorm = rawOut[3][i]
                val objConf = rawOut[4][i]

                // Ubah ke pixel
                val centerX = xNorm * originalWidth
                val centerY = yNorm * originalHeight
                val width = wNorm * originalWidth
                val height = hNorm * originalHeight

                if (objConf <= 0.3f) continue  // threshold confidence

                val left = centerX - width / 2f
                val top = centerY - height / 2f
                val right = centerX + width / 2f
                val bottom = centerY + height / 2f

                // Jika model memiliki multi‐class, rawOut akan memiliki lebih banyak baris.
                // Contoh: rawOut[5 + j][i] = class j confidence.
                // Tetapi dalam format [1,5,22869], baris sisanya tidak ada → kita asumsikan single‐class.
                val classIndex = 0
                val className = if (labels.isNotEmpty()) labels[0] else "Unknown"

                candidates.add(
                    BoundingBox(
                        x1 = left,
                        y1 = top,
                        x2 = right,
                        y2 = bottom,
                        cx = centerX,
                        cy = centerY,
                        w = width,
                        h = height,
                        cnf = objConf,
                        cls = classIndex,
                        clsName = className
                    )
                )
            }

            // Jalankan NMS dan kembalikan maksimal 1 box
            return nonMaxSuppression(candidates, iouThreshold = 0.5f, limit = 1)

        } catch (e: Exception) {
            Log.e(TAG, "Error processing raw output", e)
            return emptyList()
        }
    }

    private fun nonMaxSuppression(
        boxes: List<BoundingBox>,
        iouThreshold: Float = 0.5f,
        limit: Int = 1
    ): List<BoundingBox> {
        val selected = mutableListOf<BoundingBox>()
        val sorted = boxes.sortedByDescending { it.cnf }.toMutableList()

        while (sorted.isNotEmpty() && selected.size < limit) {
            val best = sorted.removeAt(0)
            selected.add(best)
            val it = sorted.iterator()
            while (it.hasNext()) {
                val b = it.next()
                if (iou(best, b) > iouThreshold) {
                    it.remove()
                }
            }
        }
        return selected
    }

    private fun iou(boxA: BoundingBox, boxB: BoundingBox): Float {
        val x1 = maxOf(boxA.x1, boxB.x1)
        val y1 = maxOf(boxA.y1, boxB.y1)
        val x2 = minOf(boxA.x2, boxB.x2)
        val y2 = minOf(boxA.y2, boxB.y2)

        val interW = maxOf(0f, x2 - x1)
        val interH = maxOf(0f, y2 - y1)
        val interArea = interW * interH

        val areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
        val areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)
        val union = areaA + areaB - interArea

        return if (union <= 0f) 0f else interArea / union
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * tensorWidth * tensorHeight * numChannel)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(tensorWidth * tensorHeight)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixelIndex = 0
        for (y in 0 until tensorHeight) {
            for (x in 0 until tensorWidth) {
                val pixel = intValues[pixelIndex++]
                val r = ((pixel shr 16) and 0xFF).toFloat()
                val g = ((pixel shr 8) and 0xFF).toFloat()
                val b = (pixel and 0xFF).toFloat()

                byteBuffer.putFloat(r / INPUT_STANDARD_DEVIATION)
                byteBuffer.putFloat(g / INPUT_STANDARD_DEVIATION)
                byteBuffer.putFloat(b / INPUT_STANDARD_DEVIATION)
            }
        }
        return byteBuffer
    }

    fun restart(isGpu: Boolean = false) {
        try {
            cleanup()
            setupInterpreter(isGpu)
            Log.d(TAG, "Detector restarted successfully with GPU: $isGpu")
        } catch (e: Exception) {
            Log.e(TAG, "Error restarting detector", e)
            throw e
        }
    }

    private fun cleanup() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }

    fun close() {
        cleanup()
        Log.d(TAG, "Detector closed")
    }
}
