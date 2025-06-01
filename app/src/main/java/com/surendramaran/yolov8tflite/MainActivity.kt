package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var preview: Preview? = null
    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null

    private lateinit var cameraExecutor: ExecutorService
    private var isDetectionMode = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)


        cameraExecutor = Executors.newSingleThreadExecutor()

        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        bindListeners()
    }

    private fun bindListeners() {
        binding.apply {
            // GPU toggle
            isGpu.setOnCheckedChangeListener { buttonView, isChecked ->
                cameraExecutor.submit {
                    detector?.restart(isGpu = isChecked)
                }
                if (isChecked) {
                    buttonView.setBackgroundColor(ContextCompat.getColor(baseContext, R.color.orange))
                } else {
                    buttonView.setBackgroundColor(ContextCompat.getColor(baseContext, R.color.gray))
                }
            }

            // Capture button
            captureButton.setOnClickListener {
                if (isDetectionMode) {
                    // Reset to preview mode
                    resetToPreviewMode()
                } else {
                    // Take photo and detect
                    takePhotoAndDetect()
                }
            }

            // Retake button
            retakeButton.setOnClickListener {
                resetToPreviewMode()
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageCapture = ImageCapture.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageCapture
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun takePhotoAndDetect() {
        val imageCapture = imageCapture ?: return

        binding.captureButton.isEnabled = false
        binding.progressBar.visibility = View.VISIBLE

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                    binding.captureButton.isEnabled = true
                    binding.progressBar.visibility = View.GONE
                    Toast.makeText(baseContext, "Photo capture failed", Toast.LENGTH_SHORT).show()
                }

                override fun onCaptureSuccess(image: ImageProxy) {
                    try {
                        // Convert ImageProxy to Bitmap
                        val bitmap = imageProxyToBitmap(image)

                        Log.d("ImageCapture", "Original image resolution: ${image.width} x ${image.height}")
                        Log.d("ImageCapture", "Converted bitmap resolution: ${bitmap.width} x ${bitmap.height}")
                        Log.d("ImageCapture", "Image format: ${image.format}")
                        Log.d("ImageCapture", "Image rotation: ${image.imageInfo.rotationDegrees}°")

                        // Process detection on background thread
                        cameraExecutor.execute {
                            try {
                                detector?.detect(bitmap)
                            } catch (e: Exception) {
                                Log.e(TAG, "Detection error: ${e.message}", e)
                                runOnUiThread {
                                    binding.captureButton.isEnabled = true
                                    binding.progressBar.visibility = View.GONE
                                    Toast.makeText(baseContext, "Detection failed: ${e.message}", Toast.LENGTH_LONG).show()
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Image conversion error: ${e.message}", e)
                        runOnUiThread {
                            binding.captureButton.isEnabled = true
                            binding.progressBar.visibility = View.GONE
                            Toast.makeText(baseContext, "Image processing failed: ${e.message}", Toast.LENGTH_LONG).show()
                        }
                    } finally {
                        image.close()
                    }
                }
            }
        )
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        return when (image.format) {
            ImageFormat.YUV_420_888 -> {
                // Convert YUV to RGB
                yuvToRgb(image)
            }
            ImageFormat.JPEG -> {
                // Direct conversion for JPEG
                val buffer = image.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            }
            else -> {
                // Fallback method for other formats
                val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)

                // Try to copy pixels directly
                try {
                    val buffer = image.planes[0].buffer
                    bitmap.copyPixelsFromBuffer(buffer)
                } catch (e: Exception) {
                    Log.w(TAG, "Direct pixel copy failed, using YUV conversion", e)
                    return yuvToRgb(image)
                }

                // Apply rotation if needed
                val matrix = Matrix().apply {
                    postRotate(image.imageInfo.rotationDegrees.toFloat())
                }

                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            }
        }
    }

    private fun yuvToRgb(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()

        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // Apply rotation if needed
        val matrix = Matrix().apply {
            postRotate(image.imageInfo.rotationDegrees.toFloat())
        }

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun resetToPreviewMode() {
        isDetectionMode = false
        binding.overlay.clear()
        binding.captureButton.text = "CAPTURE"
        binding.retakeButton.visibility = View.GONE
        binding.inferenceTime.visibility = View.GONE
        binding.resultText.visibility = View.GONE
        binding.captureButton.isEnabled = true
        binding.progressBar.visibility = View.GONE
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        if (it[Manifest.permission.CAMERA] == true) {
            startCamera()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        cameraExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf(
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            isDetectionMode = true
            binding.overlay.clear()
            binding.captureButton.text = "RESET"
            binding.retakeButton.visibility = View.VISIBLE
            binding.inferenceTime.visibility = View.VISIBLE
            binding.resultText.visibility = View.VISIBLE
            binding.resultText.text = "No objects detected"
            binding.captureButton.isEnabled = true
            binding.progressBar.visibility = View.GONE
        }
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long, originalImageWidth: Int, originalImageHeight: Int) {
        runOnUiThread {
            isDetectionMode = true
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes, originalImageWidth, originalImageHeight)
                invalidate()
            }
            Log.d("boundingBox", "Total detections: ${boundingBoxes.size}")
            boundingBoxes.forEachIndexed { index, box ->
                Log.d("boundingBox", "Detection $index:")
                Log.d("boundingBox", "  Class: ${box.clsName} (${box.cls})")
                Log.d("boundingBox", "  Confidence: ${box.cnf}")
                Log.d(
                    "boundingBox",
                    "  Coordinates: x1=${box.x1}, y1=${box.y1}, x2=${box.x2}, y2=${box.y2}"
                )
                Log.d("boundingBox", "  Center: cx=${box.cx}, cy=${box.cy}")
                Log.d("boundingBox", "  Size: w=${box.w}, h=${box.h}")
                Log.d("boundingBox", "  ----")
            }
            binding.captureButton.text = "RESET"
            binding.retakeButton.visibility = View.VISIBLE
            binding.inferenceTime.visibility = View.VISIBLE
            binding.resultText.visibility = View.VISIBLE
            binding.resultText.text = "Detected ${boundingBoxes.size} object(s)"
            binding.captureButton.isEnabled = true
            binding.progressBar.visibility = View.GONE
            Log.d(TAG, "Original image for Overlay: ${originalImageWidth}x${originalImageHeight}")
            for (box in boundingBoxes) {
                Log.d(
                    TAG,
                    "  Overlay Input Box: Label=${box.clsName}, Coords=[${box.x1}, ${box.y1}, ${box.x2}, ${box.y2}]"
                )
            }
        }
    }
}
