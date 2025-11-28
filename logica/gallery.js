export function addToGallery(imageCanvas, label, score, gallery) {
    const div = document.createElement('div');
    div.className = 'col';
    div.innerHTML = `
        <div class="card h-100">
            <img src="${imageCanvas.toDataURL()}" class="card-img-top" alt="Detected person">
            <div class="card-body d-flex flex-column justify-content-center align-items-start" style="height:70px;">
                <h5 class="card-title">${label}</h5>
                <p class="card-text">Score: ${(score*100).toFixed(1)}%</p>
            </div>
        </div>
    `;
    gallery.prepend(div);
    while (gallery.children.length > 8) gallery.removeChild(gallery.lastChild);
}
