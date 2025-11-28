export async function loadPoseModel() {
    const poseModel = await tf.loadLayersModel('pose_clasification_model_tfjs/model.json');
    const labelsResp = await fetch('pose_clasification_model_tfjs/pose_clasification_model_labels.json');
    const poseLabels = await labelsResp.json();
    return { poseModel, poseLabels };
}

export async function predictPoseFromLandmarks(poseModel, poseLabels, landmarks) {
    const inputArr = [];
    for (const lm of landmarks) {
        inputArr.push(lm.x, lm.y, lm.z, lm.visibility ?? 1.0);
    }
    const input = tf.tensor2d([inputArr], [1, inputArr.length]);
    const prediction = poseModel.predict(input);
    const predArr = await prediction.data();
    const maxIdx = predArr.indexOf(Math.max(...predArr));
    const label = poseLabels[maxIdx];
    const score = predArr[maxIdx];
    input.dispose();
    prediction.dispose();
    return { label, score };
}
