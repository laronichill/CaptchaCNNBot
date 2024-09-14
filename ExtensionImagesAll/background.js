chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.message) {
        chrome.notifications.create({
            type: "basic",
            iconUrl: "icon.png", // Make sure you have an icon.png file in your extension directory
            title: "Image Saved",
            message: request.message
        });
    }
});
