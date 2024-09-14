let clickedImageSrc = null;

// Listen for clicks on images
document.addEventListener("click", function (e) {
    if (e.target.tagName.toLowerCase() === "img") {
        clickedImageSrc = e.target.src;
    }
});

// Function to set up the Enter key listener
function setupEnterKeyListener() {
    const inputElement = document.getElementById("usersol");
    if (inputElement) {
        inputElement.addEventListener("keydown", function (e) {
            if (e.key === "Enter" && clickedImageSrc) {
                const inputText = e.target.value;
                if (inputText.length === 3) {
                    fetch(clickedImageSrc)
                        .then(response => response.blob())
                        .then(blob => {
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = `${inputText}.png`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                            
                            // Send a notification
                            chrome.runtime.sendMessage({message: "Image saved as " + inputText + ".png"});
                        });
                }
            }
        });
    } else {
        // Retry after a short delay if the element is not found
        setTimeout(setupEnterKeyListener, 100);
    }
}

// Start by trying to set up the event listener
setupEnterKeyListener();
