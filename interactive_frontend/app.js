// ============================================================================
// MODEL CONFIGURATION
// ============================================================================
// Adjust these values to match your model's requirements:
// - MODEL_INPUT_WIDTH/HEIGHT: The size your model expects for input images
// - CONFIDENCE_THRESHOLD_PERCENT: Minimum confidence to show detections (0-100)
// - INFERENCE_SKIP_FRAMES: Run inference every N frames (higher = faster, lower latency)
// ============================================================================
let MODEL_INPUT_WIDTH = 640;
let MODEL_INPUT_HEIGHT = 480;
const CONFIDENCE_THRESHOLD_PERCENT = 50.0; // Minimum confidence in % (0-100)

// ============================================================================
// PERFORMANCE SETTINGS (adjustable via UI)
// ============================================================================
let INFERENCE_SKIP_FRAMES = 2;  // Run inference every N frames (1 = every frame, 2 = every other frame)
let TARGET_MODEL_FPS = 15;      // Target inference FPS (lower = faster rendering)
let MIN_INFERENCE_INTERVAL = 1000 / TARGET_MODEL_FPS;  // Minimum ms between inferences

// Configure WASM backend for optimal performance
ort.env.wasm.numThreads = 4;  // Use 4 threads for parallel processing
ort.env.wasm.simd = true;     // Enable SIMD for faster operations

// Global state
let stream = null;
let session = null;
let currentCameraIndex = 0;
let availableCameras = [];
let isProcessing = false;
let lastDetections = [];
let hasUserSwitchedCamera = false;  // Track if user manually switched camera

// FPS counters
let frameCount = 0;
let lastFrameTime = performance.now();
let modelFrameCount = 0;
let lastModelTime = performance.now();
let lastInferenceTime = 0;
let inferenceFrameCounter = 0;

// Reusable canvas for performance (avoid creating new canvas every frame)
let tempCanvas = null;
let tempCtx = null;

// Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const flipCameraBtn = document.getElementById('flipCamera');
// const fullscreenBtn = document.getElementById('toggleFullscreen');
const toggleSettingsBtn = document.getElementById('toggleSettings');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const modelFpsEl = document.getElementById('modelFps');
const detectionsEl = document.getElementById('detections');
const modelUploadPanel = document.getElementById('modelUpload');
const modelFileInput = document.getElementById('modelFile');
const uploadStatusEl = document.getElementById('uploadStatus');
const loadDefaultModelBtn = document.getElementById('loadDefaultModel');
const settingsPanel = document.getElementById('settingsPanel');
const closeSettingsBtn = document.getElementById('closeSettings');
const applySettingsBtn = document.getElementById('applySettings');
const inputResolutionSelect = document.getElementById('inputResolution');
const inferenceRateSelect = document.getElementById('inferenceRate');
const skipFramesSelect = document.getElementById('skipFrames');

// Target FPS for video rendering
const TARGET_FPS = 24;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

// Initialize the application
async function init() {
    try {
        console.log('Using ONNX Runtime with WASM backend (CPU)');
        console.log('WASM Config: numThreads=4, SIMD=enabled');
        
        // Get available cameras
        await enumerateCameras();
        
        // Start webcam
        await startCamera();
        
        // Setup event listeners
        flipCameraBtn.addEventListener('click', switchCamera);
        // fullscreenBtn.addEventListener('click', toggleFullscreen);
        toggleSettingsBtn.addEventListener('click', toggleSettings);
        closeSettingsBtn.addEventListener('click', closeSettings);
        applySettingsBtn.addEventListener('click', applySettings);
        modelFileInput.addEventListener('change', handleModelUpload);
        loadDefaultModelBtn.addEventListener('click', loadDefaultModel);
        
        // Start rendering loop
        requestAnimationFrame(renderLoop);
        
    } catch (error) {
        console.error('Initialization error:', error);
        statusEl.textContent = 'Error: ' + error.message;
        statusEl.className = 'error';
    }
}

// Enumerate available cameras
async function enumerateCameras() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    availableCameras = devices.filter(device => device.kind === 'videoinput');
    
    if (availableCameras.length <= 1) {
        flipCameraBtn.disabled = true;
    }
}

// Start camera stream
async function startCamera() {
    // Stop existing stream if any
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    // Build video constraints
    // Default to environment (rear) camera for mobile devices
    // Only use specific deviceId when user has manually switched cameras
    const videoConstraints = {
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: TARGET_FPS }
    };
    
    if (hasUserSwitchedCamera && availableCameras[currentCameraIndex]?.deviceId) {
        // User manually switched camera - use specific deviceId
        videoConstraints.deviceId = { exact: availableCameras[currentCameraIndex].deviceId };
    } else {
        // Default to environment (rear) camera on mobile devices
        videoConstraints.facingMode = { ideal: 'environment' };
    }
    
    const constraints = { video: videoConstraints };
    
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    
    // Wait for video to be ready
    await new Promise(resolve => {
        video.onloadedmetadata = () => {
            resizeCanvas();
            resolve();
        };
    });
}

// Switch between cameras
async function switchCamera() {
    hasUserSwitchedCamera = true;  // User manually switching - use specific deviceId
    currentCameraIndex = (currentCameraIndex + 1) % availableCameras.length;
    await startCamera();
}

// Resize canvas to match video
function resizeCanvas() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

// Toggle fullscreen
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
        document.body.classList.add('fullscreen');
    } else {
        document.exitFullscreen();
        document.body.classList.remove('fullscreen');
    }
}

// Toggle settings panel
function toggleSettings() {
    settingsPanel.classList.toggle('hidden');
}

// Close settings panel
function closeSettings() {
    settingsPanel.classList.add('hidden');
}

// Apply performance settings
function applySettings() {
    // Get selected resolution
    const resolution = inputResolutionSelect.value.split(',');
    MODEL_INPUT_WIDTH = parseInt(resolution[0]);
    MODEL_INPUT_HEIGHT = parseInt(resolution[1]);
    
    // Get inference rate
    TARGET_MODEL_FPS = parseInt(inferenceRateSelect.value);
    MIN_INFERENCE_INTERVAL = 1000 / TARGET_MODEL_FPS;
    
    // Get frame skip
    INFERENCE_SKIP_FRAMES = parseInt(skipFramesSelect.value);
    
    // Reset temp canvas to force recreation with new size
    tempCanvas = null;
    tempCtx = null;
    
    // Reset inference counters
    lastInferenceTime = 0;
    inferenceFrameCounter = 0;
    
    console.log('Performance settings applied:');
    console.log(`- Input Resolution: ${MODEL_INPUT_WIDTH}x${MODEL_INPUT_HEIGHT}`);
    console.log(`- Target Model FPS: ${TARGET_MODEL_FPS}`);
    console.log(`- Frame Skip: Every ${INFERENCE_SKIP_FRAMES} frame(s)`);
    
    // Show feedback
    statusEl.textContent = `Settings Applied: ${MODEL_INPUT_WIDTH}x${MODEL_INPUT_HEIGHT} @ ${TARGET_MODEL_FPS} FPS`;
    setTimeout(() => {
        if (session) {
            statusEl.textContent = 'Model Ready (WASM)';
        }
    }, 2000);
    
    closeSettings();
}

// Load default model from model-s folder
async function loadDefaultModel() {
    try {
        uploadStatusEl.textContent = 'Loading default model...';
        statusEl.textContent = 'Loading default model...';
        loadDefaultModelBtn.disabled = true;
        
        // Fetch the default model
        const response = await fetch('model-s/best.onnx');
        if (!response.ok) {
            throw new Error('Could not load default model. Make sure model-s/best.onnx exists.');
        }
        
        const arrayBuffer = await response.arrayBuffer();
        await loadModelFromBuffer(arrayBuffer, 'Default Model (model-s)');
        
    } catch (error) {
        console.error('Error loading default model:', error);
        uploadStatusEl.textContent = 'Error: ' + error.message;
        statusEl.textContent = 'Model Error';
        statusEl.className = 'error';
        loadDefaultModelBtn.disabled = false;
    }
}

// Handle model file upload
async function handleModelUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
        uploadStatusEl.textContent = 'Loading model...';
        statusEl.textContent = 'Loading model...';
        
        // Read file as array buffer
        const arrayBuffer = await file.arrayBuffer();
        await loadModelFromBuffer(arrayBuffer, file.name);
        
    } catch (error) {
        console.error('Model loading error:', error);
        uploadStatusEl.textContent = 'Error loading model: ' + error.message;
        statusEl.textContent = 'Model Error';
        statusEl.className = 'error';
    }
}

// Common function to load model from array buffer
async function loadModelFromBuffer(arrayBuffer, modelName) {
    // Create ONNX Runtime session with WASM backend
    console.log('Loading model with WASM backend...');
    session = await ort.InferenceSession.create(arrayBuffer, {
        executionProviders: ['wasm'], 
        graphOptimizationLevel: 'all'
    });
    
    console.log('====================================');
    console.log('MODEL LOADED SUCCESSFULLY');
    console.log('====================================');
    console.log('Model:', modelName);
    console.log('Input names:', session.inputNames);
    console.log('Output names:', session.outputNames);
    console.log('Execution Provider: WASM (CPU)');
    
    // Try to get more info about inputs/outputs
    try {
        const inputMetadata = session.inputNames.map(name => ({
            name: name,
            // Note: ONNX Runtime Web may not expose all metadata
        }));
        const outputMetadata = session.outputNames.map(name => ({
            name: name,
        }));
        console.log('Input metadata:', inputMetadata);
        console.log('Output metadata:', outputMetadata);
    } catch (e) {
        console.log('(Detailed metadata not available)');
    }
    console.log('====================================\n');
    
    uploadStatusEl.textContent = `Model loaded: ${modelName} (WASM)`;
    statusEl.textContent = 'Model Ready (WASM)';
    statusEl.className = 'ready';
    
    // Hide upload panel
    setTimeout(() => {
        modelUploadPanel.classList.add('hidden');
    }, 1000);
}

// Main rendering loop
let lastRenderTime = 0;
function renderLoop(timestamp) {
    // Throttle to target FPS
    if (timestamp - lastRenderTime >= FRAME_INTERVAL) {
        // Update FPS counter
        frameCount++;
        const elapsed = timestamp - lastFrameTime;
        if (elapsed >= 1000) {
            fpsEl.textContent = `FPS: ${Math.round(frameCount * 1000 / elapsed)}`;
            frameCount = 0;
            lastFrameTime = timestamp;
        }
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw detections
        drawDetections();
        
        // Run model inference with frame skipping and time throttling
        inferenceFrameCounter++;
        const timeSinceLastInference = timestamp - lastInferenceTime;
        
        if (session && !isProcessing && 
            inferenceFrameCounter >= INFERENCE_SKIP_FRAMES &&
            timeSinceLastInference >= MIN_INFERENCE_INTERVAL) {
            inferenceFrameCounter = 0;
            lastInferenceTime = timestamp;
            runInference();
        }
        
        lastRenderTime = timestamp;
    }
    
    requestAnimationFrame(renderLoop);
}

// Run model inference asynchronously
async function runInference() {
    isProcessing = true;
    
    try {
        // Capture current frame
        const imageData = captureFrame();
        if (!imageData) {
            isProcessing = false;
            return;
        }
        
        // Preprocess image for model
        const inputTensor = preprocessImage(imageData);
        
        // Run inference
        const feeds = {};
        feeds[session.inputNames[0]] = inputTensor;
        
        const results = await session.run(feeds);

        // Process results
        const outputTensor = results[session.outputNames[0]];

        lastDetections = processDetections(outputTensor);
        
        // Update model FPS
        modelFrameCount++;
        const elapsed = performance.now() - lastModelTime;
        if (elapsed >= 1000) {
            modelFpsEl.textContent = `Model FPS: ${Math.round(modelFrameCount * 1000 / elapsed)}`;
            modelFrameCount = 0;
            lastModelTime = performance.now();
        }
        
        detectionsEl.textContent = `Detections: ${lastDetections.length}`;
        
    } catch (error) {
        console.error('Inference error:', error);
    } finally {
        isProcessing = false;
    }
}

// Capture frame from video (optimized with reusable canvas)
function captureFrame() {
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        return null;
    }
    
    // Reuse canvas to avoid allocation overhead
    if (!tempCanvas) {
        tempCanvas = document.createElement('canvas');
        tempCanvas.width = MODEL_INPUT_WIDTH;
        tempCanvas.height = MODEL_INPUT_HEIGHT;
        tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
    }
    
    // Draw video frame to temp canvas (resized to model input size)
    tempCtx.drawImage(video, 0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
    
    return tempCtx.getImageData(0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
}

// Preprocess image for model input (optimized)
function preprocessImage(imageData) {
    const { width, height, data } = imageData;
    const totalPixels = width * height;
    
    // Convert to normalized RGB float array [1, 3, height, width]
    const float32Data = new Float32Array(3 * totalPixels);
    
    // Normalize and convert from RGBA to RGB (CHW format)
    // Using separate loops can be faster due to better cache locality
    const stride2 = totalPixels * 2;
    for (let i = 0, j = 0; i < totalPixels; i++, j += 4) {
        float32Data[i] = data[j] * 0.00392156862745098; // R (1/255)
        float32Data[totalPixels + i] = data[j + 1] * 0.00392156862745098; // G
        float32Data[stride2 + i] = data[j + 2] * 0.00392156862745098; // B
    }
    
    return new ort.Tensor('float32', float32Data, [1, 3, height, width]);
}

// Process model output to get detections (optimized)
function processDetections(outputTensor) {
    const detections = [];
    const data = outputTensor.data;
    const dims = outputTensor.dims;
    
    if (dims.length === 3) {
        const numDetections = dims[1];
        const numFeatures = dims[2];
        const confidenceThreshold = 0.5;
        
        // Early exit if no detections
        if (numDetections === 0) return detections;
        
        for (let i = 0; i < numDetections; i++) {
            const offset = i * numFeatures;
            const confidence = data[offset + 4];
            
            // Skip low-confidence detections early
            if (confidence < confidenceThreshold) continue;
            
            // Create detection object
            detections.push({
                x1: data[offset],
                y1: data[offset + 1],
                x2: data[offset + 2],
                y2: data[offset + 3],
                confidence: confidence,
                keypoints: [
                    { x: data[offset + 6], y: data[offset + 7] },
                    { x: data[offset + 8], y: data[offset + 9] },
                    { x: data[offset + 10], y: data[offset + 11] },
                    { x: data[offset + 12], y: data[offset + 13] },
                    { x: data[offset + 14], y: data[offset + 15] },
                    { x: data[offset + 16], y: data[offset + 17] },
                    { x: data[offset + 18], y: data[offset + 19] }
                ]
            });
        }
    } else {
        console.warn(`⚠️ Unexpected tensor dimensions: ${dims.length}D tensor`);
        console.log('Full dims:', dims);
        console.log('You may need to customize the processDetections function for your model');
    }
    
    return detections;
}

// Draw detections on canvas
function drawDetections() {
    if (lastDetections.length === 0) return;
    
    // Scale factors to map from model input size to canvas size
    const scaleX = canvas.width / MODEL_INPUT_WIDTH;
    const scaleY = canvas.height / MODEL_INPUT_HEIGHT;
    
    // Keypoint colors (matching your rendering script)
    const keypointColors = [
        [4, 44, 235],      // hub_rear (blue)
        [13, 255, 0],      // tower_top (green)
        [238, 234, 242],   // tower_bottom (light)
        [8, 217, 182],     // hub_front (cyan)
        [241, 117, 224],   // tip1 (pink)
        [248, 69, 91],     // tip2 (red)
        [19, 28, 93]       // tip3 (dark blue)
    ];
    
    lastDetections.forEach((det, idx) => {
        // Scale bounding box coordinates
        const x1 = det.x1 * scaleX;
        const y1 = det.y1 * scaleY;
        const x2 = det.x2 * scaleX;
        const y2 = det.y2 * scaleY;
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Draw bounding box
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw keypoints if available
        if (det.keypoints && det.keypoints.length > 0) {
            det.keypoints.forEach((kp, kpIdx) => {
                const kp_x = kp.x * scaleX;
                const kp_y = kp.y * scaleY;
                
                // Use corresponding color from the color map
                const color = keypointColors[kpIdx % keypointColors.length];
                ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                
                // Draw keypoint circle
                ctx.beginPath();
                ctx.arc(kp_x, kp_y, 6, 0, 2 * Math.PI);
                ctx.fill();
                
                // Draw white border around keypoint
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
        }
        
        // Draw label background
        ctx.font = '16px Arial';
        const confidencePercent = det.confidence > 1 ? det.confidence : det.confidence * 100;
        const label = `Wind Turbine ${confidencePercent.toFixed(1)}%`;
        const textMetrics = ctx.measureText(label);
        const textHeight = 20;
        
        ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
        ctx.fillRect(x1, y1 - textHeight - 4, textMetrics.width + 10, textHeight + 4);
        
        // Draw label text
        ctx.fillStyle = '#000';
        ctx.fillText(label, x1 + 5, y1 - 8);
    });
}

// Handle window resize
window.addEventListener('resize', () => {
    if (video.videoWidth > 0) {
        resizeCanvas();
    }
});

// Start the application
init();

