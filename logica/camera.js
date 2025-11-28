export async function setupCamera(video, cameraInstructions) {
    let cameraStream = null;
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 800, height: 640 } });
        video.srcObject = cameraStream;
        await video.play();
        cameraInstructions.style.display = 'none';
        return cameraStream;
    } catch (err) {
        showCameraInstructions(cameraInstructions, 'No se pudo acceder a la cámara.<br>Permite el acceso o revisa la configuración del navegador.', 'danger');
        throw err;
    }
}

export function stopCamera(video, cameraStream, cameraInstructions, toggleCameraBtn) {
    if (cameraStream) {
        const tracks = cameraStream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
        cameraInstructions.style.display = '';
        cameraInstructions.innerHTML = 'Cámara desactivada. Pulsa F5 para reactivar.';
    }
    if (toggleCameraBtn) {
        toggleCameraBtn.textContent = 'Activar cámara';
    }
}

export function showCameraInstructions(cameraInstructions, msg, type = 'info') {
    cameraInstructions.style.display = '';
    cameraInstructions.className = 'alert alert-' + type + ' mt-3 w-100 text-center';
    cameraInstructions.innerHTML = msg;
}
