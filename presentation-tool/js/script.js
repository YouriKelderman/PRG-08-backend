import {FilesetResolver, HandLandmarker} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

document.addEventListener('DOMContentLoaded', init);

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;

let handSign;
let canvasWidth;
let canvasHeight;
let presentationID;
let lastVideoTime = -1;
let results = undefined;
let pose;
let sortedResults;
let currentSlide = 1;
let lastCircleSpawnTime = performance.now();
const circleSpawnInterval = 1000;

let networkPredictions = document.getElementById('predictions')

let video = document.getElementById("webcam");
let canvasElement = document.getElementById("output_canvas");
let canvasCtx = canvasElement.getContext("2d");


const nn = ml5.neuralNetwork({task: 'classification', debug: true})
const modelDetails = {
    model: '../model/model.json',
    metadata: '../model/model_meta.json',
    weights: '../model/model.weights.bin'
}

function init() {
    createHandLandmarker();

    const hasGetUserMedia = () => {
        let _a;
        return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia);
    };
    handSign = document.getElementById("handBox");
    if (hasGetUserMedia()) {
        enableWebcamButton = document.getElementById("webcamButton");
        enableWebcamButton.addEventListener("click", enableCam);
    } else {
        console.warn("getUserMedia() is not supported by your browser");
    }
    nn.load(modelDetails, () => console.log("Finished Loading..."))

    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);
}

const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 1
    });
};


function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "Start Tracking";
    } else {

        webcamRunning = true;
        enableWebcamButton.innerText = "Stop Tracking";
    }

    const constraints = {
        video: true
    };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}


function setCanvasSize() {
    canvasElement.style.width = window.innerWidth + 'px';
    canvasElement.style.height = window.innerHeight + 'px';
    canvasElement.width = window.innerWidth;
    canvasElement.height = window.innerHeight;
    canvasWidth = canvasElement.width;
    canvasHeight = canvasElement.height;
}

let lastPrediction;

async function predictWebcam() {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({runningMode: "VIDEO"});
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
    }

    canvasCtx.save();
    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            pose = landmarks.flatMap(coord => [coord.x, coord.y, coord.z]);
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#01ffff",
                lineWidth: 1
            });

            drawLandmarks(canvasCtx, landmarks, {color: "#00c5ff", lineWidth: 1});

            if (pose !== undefined) {
                canvasCtx.beginPath();
                canvasCtx.arc(pose[24] * canvasWidth, pose[25] * canvasHeight, 5, 0, Math.PI * 2);
                canvasCtx.fillStyle = 'blue';
                canvasCtx.fill();
                canvasCtx.closePath();
            }

            const currentTime = performance.now();
            if (currentTime - lastCircleSpawnTime >= circleSpawnInterval) {
                createCircle();
                lastCircleSpawnTime = currentTime;
            }


            let predictions = await nn.classify(pose);
            sortedResults = predictions.sort((a, b) => b.confidence - a.confidence);

            let fragment = document.createDocumentFragment();

            sortedResults.forEach(prediction => {
                let p = document.createElement('p');
                p.innerText = `${prediction.label}: ${prediction.confidence}`;
                fragment.appendChild(p);
            });

            if (lastPrediction !== sortedResults[0].label) {

                if (sortedResults[0].label === "okay") {
                    handSign.style.opacity = "0";
                    currentSlide++;
                    // Get a reference to the iframe
                    let url = `https://docs.google.com/presentation/d/e/${presentationID}/embed?start=true&loop=false&delayms=300000&slide=${currentSlide}`;
                    document.getElementById('presentation').setAttribute("src", url);
                    document.getElementById('presentation').style.opacity = "1";
                } else if (sortedResults[0].label === "open") {
                    handSign.style.opacity = "1";
                }
                else{
                    handSign.style.opacity = "0";
                }
                lastPrediction = sortedResults[0].label;
            }

            networkPredictions.innerHTML = '';
            networkPredictions.appendChild(fragment);
        }
    }
    canvasCtx.restore();
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

document.addEventListener("click", function (event) {
    if (event.target && event.target.matches("input[type='submit']")) {
        presentationID = document.getElementById("link").value;
        if (presentationID.trim() !== "") {
            presentationID = document.getElementById("link").value;
            let url = `https://docs.google.com/presentation/d/e/${presentationID}/embed?start=true&loop=false&delayms=300000&slide=0`;
            document.getElementById('presentation').setAttribute("src", url);
            document.getElementById('presentation').style.opacity = "1";
        } else {
            alert("Please enter a presentation link.");
        }
    }
});

function createCircle() {
    const radius = Math.random() * 20 + 10;
    const x = Math.random() * (canvasElement.width - 2 * radius) + radius;
    const y = Math.random() * (canvasElement.height - 2 * radius) + radius;
    const dx = (Math.random() - 0.5) * 2;
    const dy = (Math.random() - 0.5) * 2;

    let circleColor;
    if (Math.random() < 0.2) {
        circleColor = '#be2525';
    } else {
        circleColor = '#2596be';
    }
    const color = circleColor


}


function stopGame() {
    webcamRunning = false;
}



