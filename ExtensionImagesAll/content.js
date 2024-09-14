let clickedImageSrc = null;

// Listen for clicks on images and store the clicked image's source
document.addEventListener("click", function (e) {
    if (e.target.tagName.toLowerCase() === "img") {
        clickedImageSrc = e.target.src;
        console.log("Clicked image source:", clickedImageSrc);  // Debugging
    }
});

// Helper function to determine if an element is above another element
function isAbove(imgElement, referenceElement) {
    const imgRect = imgElement.getBoundingClientRect();
    const refRect = referenceElement.getBoundingClientRect();
    return imgRect.bottom < refRect.top;
}

// Helper function to convert base64 to Blob
function base64ToBlob(base64, mime) {
    const byteCharacters = atob(base64.split(',')[1]);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mime });
}

// Function to download base64 image
function downloadBase64Image(src, name) {
    if (src.startsWith('data:image')) {
        const mime = src.split(';')[0].split(':')[1];
        const blob = base64ToBlob(src, mime);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log("Download triggered for:", name);  // Debugging
    } else {
        console.error("Not a valid base64 image.");
    }
}

// Collect images from the specified div that are above another div
function collectImageSrcs() {
    const containerDiv = document.getElementById("div");
    const referenceDiv = document.getElementById("sub");
    let imageSrcList = [];

    if (containerDiv && referenceDiv) {
        const images = containerDiv.querySelectorAll("img");
        imageSrcList = Array.from(images).filter(img => {
            return isAbove(img, referenceDiv) && img.src !== clickedImageSrc;
        }).map(img => img.src);
        console.log("Collected image srcs:", imageSrcList);
    } else {
        console.error("Container or reference div not found.");
    }

    return imageSrcList;
}

// Function to handle image saving on Enter key press
function handleImageSaving() {
    const inputElement = document.getElementById("usersol");
    if (!inputElement) {
        console.error("Input element not found.");
        return;
    }

    let inputText = inputElement.value.trim().toUpperCase();  // Auto-capitalize and trim spaces
    if (inputText.length !== 3) {
        console.warn("Input text length is not 3 characters.");
        return;
    }

    if (clickedImageSrc) {
        console.log("Starting download for clicked image:", clickedImageSrc);

        // Save the clicked image with the name XXX_0.png
        downloadBase64Image(clickedImageSrc, `${inputText}_0.png`);

        // Collect other images and save them sequentially
        const imageSrcList = collectImageSrcs();
        imageSrcList.slice(0, 7).forEach((src, index) => {
            const fileName = `${inputText}_${index + 1}.png`;
            downloadBase64Image(src, fileName);
        });

        console.log("All images saved.");
    } else {
        console.warn("No image clicked.");
    }
}

// Set up keydown event listener
function setupKeyListener() {
    document.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {  // Trigger download on Enter key press
            console.log("Enter key pressed");
            handleImageSaving();
        }
    });
}

// Initialize script after page load
window.addEventListener('load', setupKeyListener);
