package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.FileInputStream
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

    private val INPUT_MEAN = 0f
    private val INPUT_STANDARD_DEVIATION = 255f

    companion object {
        private const val TAG = "Detector"
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    init {
        try {
            setupInterpreter(useGpu = false) // Start with CPU by default for stability
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

            // Get input tensor dimensions
            val inputShape = interpreter!!.getInputTensor(0).shape()
            tensorHeight = inputShape[1]
            tensorWidth = inputShape[2]
            numChannel = if (inputShape.size > 3) inputShape[3] else 3
            numElements = tensorHeight * tensorWidth * numChannel

            Log.d(TAG, "Model input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "Tensor dimensions: ${tensorWidth}x${tensorHeight}x${numChannel}")

            // Debug output tensor shape
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            Log.d(TAG, "Model output shape: ${outputShape.contentToString()}")

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
            Log.d(TAG, "Loaded ${labels.size} labels: $labels")

            if (labels.isEmpty()) {
                throw IllegalStateException("No labels found in $labelPath")
            }
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

            // Resize bitmap to model input size
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, tensorWidth, tensorHeight, false)

            // Prepare input tensor
            val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

            // Prepare output tensor
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            val outputBuffer = Array(1) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }

            // Run inference
            interpreter!!.run(inputBuffer, outputBuffer)

            val inferenceTime = System.currentTimeMillis() - startTime

            // Debug output dimensions
            Log.d(TAG, "Output buffer dimensions: ${outputBuffer.size} x ${outputBuffer[0].size} x ${outputBuffer[0][0].size}")

            // Process results
            val boundingBoxes = processOutput(outputBuffer[0], bitmap.width, bitmap.height)

            if (boundingBoxes.isEmpty()) {
                detectorListener.onEmptyDetect()
            } else {
                detectorListener.onDetect(boundingBoxes, inferenceTime)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error during detection", e)
            detectorListener.onEmptyDetect()
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val inputBuffer = Array(1) { Array(tensorHeight) { Array(tensorWidth) { FloatArray(3) } } }

        for (y in 0 until tensorHeight) {
            for (x in 0 until tensorWidth) {
                val pixel = bitmap.getPixel(x, y)

                // Normalize pixel values
                inputBuffer[0][y][x][0] = ((pixel shr 16 and 0xFF) - INPUT_MEAN) / INPUT_STANDARD_DEVIATION
                inputBuffer[0][y][x][1] = ((pixel shr 8 and 0xFF) - INPUT_MEAN) / INPUT_STANDARD_DEVIATION
                inputBuffer[0][y][x][2] = ((pixel and 0xFF) - INPUT_MEAN) / INPUT_STANDARD_DEVIATION
            }
        }

        return inputBuffer
    }

    private fun processOutput(output: Array<FloatArray>, originalWidth: Int, originalHeight: Int): List<BoundingBox> {
        val boundingBoxes = mutableListOf<BoundingBox>()

        try {
            Log.d(TAG, "Processing output with ${output.size} detections")

            for (i in output.indices) {
                val detection = output[i]
                Log.d(TAG, "Detection $i: size=${detection.size}, values=${detection.take(10).toString()}...")

                // Validate detection array size
                if (detection.size < 5) {
                    Log.w(TAG, "Detection $i has insufficient data: ${detection.size} elements")
                    continue
                }

                val confidence = detection[4]
                Log.d(TAG, "Detection $i confidence: $confidence")

                if (confidence > 0.3f) { // Confidence threshold
                    val centerX = detection[0] * originalWidth
                    val centerY = detection[1] * originalHeight
                    val width = detection[2] * originalWidth
                    val height = detection[3] * originalHeight

                    Log.d("boundingBox", "ini originalWidth: ${originalWidth} originalHeight: ${originalHeight}")
                    Log.d("boundingBox", "ini width: ${width} height: ${height}")

                    var left = centerX - originalWidth / 2
                    var top = centerY - originalHeight / 2
                    var right = centerX + originalWidth / 2
                    var bottom = centerY + originalHeight / 2

                    left = left.coerceIn(0f, originalWidth.toFloat())
                    top = top.coerceIn(0f, originalHeight.toFloat())
                    right = right.coerceIn(0f, originalWidth.toFloat())
                    bottom = bottom.coerceIn(0f, originalHeight.toFloat())

                    // Opsional: Jika setelah clipping, lebar atau tinggi menjadi <= 0, abaikan deteksi ini
                    if (right <= left || bottom <= top) {
                         Log.d(TAG, "Detection $i discarded after clipping (zero or negative size)")
                        continue
                    }

                    val classIndex = if (detection.size > 5) {
                        // Multi-class: find class with highest probability
                        var maxIndex = 5
                        val maxSearchIndex = minOf(detection.size - 1, 5 + labels.size - 1)

                        for (j in 6..maxSearchIndex) {
                            if (j < detection.size && detection[j] > detection[maxIndex]) {
                                maxIndex = j
                            }
                        }
                        maxIndex - 5
                    } else {
                        0 // Single class
                    }

                    // Validate class index
                    val safeClassIndex = if (classIndex >= 0 && classIndex < labels.size) {
                        classIndex
                    } else {
                        Log.w(TAG, "Invalid class index: $classIndex, using 0")
                        0
                    }

                    val className = if (safeClassIndex < labels.size) {
                        labels[safeClassIndex]
                    } else {
                        "Unknown"
                    }

                    Log.d(TAG, "Adding bounding box: class=$className, confidence=$confidence")

                    boundingBoxes.add(
                        BoundingBox(
                            x1 = left,
                            y1 = top,
                            x2 = right,
                            y2 = bottom,
                            cx = centerX,
                            cy = centerY,
                            w = width,
                            h = height,
                            cnf = confidence,
                            cls = safeClassIndex,
                            clsName = className
                        )
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing output", e)
        }

        Log.d(TAG, "Found ${boundingBoxes.size} valid detections")
        return boundingBoxes
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