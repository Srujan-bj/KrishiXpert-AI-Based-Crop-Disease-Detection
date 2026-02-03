const BASE_URL = "https://krishixpert-ai-based-crop-disease.onrender.com";

async function predictDisease() {

    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload an image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {

        const response = await fetch(`${BASE_URL}/predict`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // =========================
        // TEXT RESULTS
        // =========================
        document.getElementById("predictionResult").innerHTML = `
            <h3>Prediction: ${data.prediction}</h3>
            <p>Confidence: ${data.confidence}%</p>
            <p>Loss: ${data.loss}%</p>
            <p>Healthy: ${data.healthy_percentage}%</p>
            <p>Unhealthy: ${data.unhealthy_percentage}%</p>
        `;

        // =========================
        // IMAGE HELPER
        // =========================
        function addImage(containerId, src) {
            if (!src) return;

            const container = document.getElementById(containerId);
            const img = document.createElement("img");

            img.src = src;
            img.width = 220;
            img.style.margin = "10px";
            img.style.boxShadow = "0 2px 10px green";

            container.appendChild(img);
        }

        // clear containers
        ["image-container","img2","img3","img4","img5","img6","img7"]
        .forEach(id => document.getElementById(id).innerHTML = "");

        // =========================
        // ADD IMAGES
        // =========================
        addImage("image-container", data.image_black_background);
        addImage("image-container", data.image_green_white);
        addImage("image-container", data.image_white_red);

        addImage("img2", data.histogram1);
        addImage("img3", data.histogram2);
        addImage("img4", data.histogram3);

        addImage("img5", data.confidence_graph);
        addImage("img6", data.handuh_graph);
        addImage("img7", data.confusion_matrix);

        document.getElementById("result").style.display = "block";

    } catch (err) {
        console.error(err);
        alert("Prediction failed. Check console.");
    }
}
