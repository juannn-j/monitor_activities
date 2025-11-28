import { setupCamera, stopCamera, showCameraInstructions } from './camera.js';
import { createPoseInstance, getBoundingBox, cropPerson } from './mediapipe.js';
import { loadPoseModel, predictPoseFromLandmarks } from './model.js';
import { addToGallery } from './gallery.js';

const video = document.getElementById('video');
const canvas = document.getElementById('output-canvas');
const gallery = document.getElementById('gallery');
const cameraInstructions = document.getElementById('camera-instructions');
const toggleCameraBtn = document.getElementById('toggle-camera');
const togglePredictionBtn = document.getElementById('toggle-prediction');

let cameraStream = null;
let mediapipeActive = true;
let predictionActive = false;
let photoInterval = null;
let lastLandmarks = null;
let poseModel = null;
let poseLabels = null;
let mainProcessFrame = null;

async function main() {
    cameraStream = await setupCamera(video, cameraInstructions);
    const modelData = await loadPoseModel();
    poseModel = modelData.poseModel;
    poseLabels = modelData.poseLabels;

    // Botón cámara
    if (toggleCameraBtn) {
        toggleCameraBtn.addEventListener('click', () => {
            if (mediapipeActive) {
                stopCamera(video, cameraStream, cameraInstructions, toggleCameraBtn);
                mediapipeActive = false;
            } else {
                setupCamera(video, cameraInstructions).then(stream => {
                    cameraStream = stream;
                    mediapipeActive = true;
                });
                toggleCameraBtn.textContent = 'Desactivar cámara';
            }
        });
    }

    // Botón predicción
    if (togglePredictionBtn) {
        togglePredictionBtn.addEventListener('click', () => {
            predictionActive = !predictionActive;
            togglePredictionBtn.textContent = predictionActive ? 'Desactivar predicción' : 'Activar predicción';
            if (predictionActive) {
                if (!photoInterval) {
                    photoInterval = setInterval(() => {
                        if (predictionActive && video.srcObject) processPrediction();
                    }, 2000);
                }
            } else {
                if (photoInterval) {
                    clearInterval(photoInterval);
                    photoInterval = null;
                }
            }
        });
    }

    // Inicializar Mediapipe Pose
    const pose = createPoseInstance(async (results) => {
        if (results.poseLandmarks) {
            lastLandmarks = results.poseLandmarks;
            if (predictionActive && !photoInterval) {
                await processPrediction();
            }
        }
    });

    async function processFrame() {
        if (!mediapipeActive) return;
        await pose.send({ image: video });
        requestAnimationFrame(mainProcessFrame);
    }
    mainProcessFrame = processFrame;
    processFrame();

    // Predicción y galería
    async function processPrediction() {
        if (!poseModel || !video.srcObject || !lastLandmarks) return;
        const bbox = getBoundingBox(lastLandmarks);
        const cropped = cropPerson(video, bbox);
        const { label, score } = await predictPoseFromLandmarks(poseModel, poseLabels, lastLandmarks);
        addToGallery(cropped, label, score, gallery);
    }
}

window.addEventListener('DOMContentLoaded', main);
