export function createPoseInstance(onResults) {
    const pose = new window.Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });
    pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    pose.onResults(onResults);
    return pose;
}

export function getBoundingBox(landmarks) {
    const xs = landmarks.map(lm => lm.x);
    const ys = landmarks.map(lm => lm.y);
    return {
        xMin: Math.max(0, Math.min(...xs)),
        yMin: Math.max(0, Math.min(...ys)),
        xMax: Math.min(1, Math.max(...xs)),
        yMax: Math.min(1, Math.max(...ys)),
    };
}

export function cropPerson(image, bbox) {
    const [w, h] = [image.videoWidth || image.width, image.videoHeight || image.height];
    const x = Math.max(0, Math.floor(bbox.xMin * w));
    const y = Math.max(0, Math.floor(bbox.yMin * h));
    const width = Math.min(w - x, Math.floor((bbox.xMax - bbox.xMin) * w));
    const height = Math.min(h - y, Math.floor((bbox.yMax - bbox.yMin) * h));
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    tempCanvas.getContext('2d').drawImage(image, x, y, width, height, 0, 0, width, height);
    return tempCanvas;
}
