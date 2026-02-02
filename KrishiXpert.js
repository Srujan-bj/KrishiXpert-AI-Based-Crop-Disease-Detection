const BASE_URL = 'RUN THE - KrishXpert.py - file and past the url here it looks like //....';

function predictDisease() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload an image.");
        return;
    }

    // Show preview of uploaded image


    const formData = new FormData();
    formData.append('file', file);

    fetch(`${BASE_URL}/predict`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Prediction details
        document.getElementById('predictionResult').innerHTML = `
            <div class="result-card">
                <h3>Prediction: ${data.prediction}</h3>
                                
                <p>Remedies: ${data.remedies}</p>

                
            </div>
        `;
        document.getElementById('predictionResult1').innerHTML = `
            <p>Accuracy: ${data.confidence}%</p>
            <p>Loss: ${data.loss}%</p>
            <p>Healthy percentage: ${data.healthy_percentage}%</p>
            <p>Unhealthy percentage: ${data.unhealthy_percentage}%</p>
            <p>COUNT: ${data.count}</p>
                   `;
    
        // Processed Leaf Images
        const container = document.getElementById('image-container');
        container.innerHTML = '';

        const images = [
            data.image_black_background,
            data.image_green_white,
            data.image_white_red
        ];

        images.forEach((src, index) => {
            const img = document.createElement('img');
            img.src = src;
            img.style.position = 'relative';
            img.style.backgroundColor = 'rgba(255, 255, 255, 0.53)';
            img.style.padding = '10px';
            img.style.marginLeft = '20px';
            img.style.marginTop = '20px';
            img.style.borderRadius = '0px';
            img.style.boxShadow = '0 2px 10px rgb(36, 250, 3)';
            img.style.display = 'block';
            img.alt = `Image ${index + 1}`;
            img.width = 200;
            container.appendChild(img);
        });

        // Histogram Image 1
        const container2 = document.getElementById('img2');
        container2.innerHTML = '';
        const img1 = document.createElement('img');
        img1.src = data.histogram1;
        img1.style.position = 'relative';
        img1.style.backgroundColor = 'rgba(255, 255, 255, 0.53)';
        img1.style.padding = '10px';
        img1.style.marginLeft = '90px';
        img1.style.marginTop = '20px';
        img1.style.borderRadius = '0px';
        img1.style.boxShadow = '0 2px 10px rgb(36, 250, 3)';
        img1.style.display = 'block';
        img1.alt = 'Histogram 1';
        img1.width = 260;
        container2.appendChild(img1);

        // Histogram Image 2
        const container3 = document.getElementById('img3');
        container3.innerHTML = '';
        const img2 = document.createElement('img');
        img2.src = data.histogram2;
        img2.style.position = 'absolute';
        img2.style.backgroundColor = 'rgba(255, 255, 255, 0.53)';
        img2.style.padding = '10px';
        img2.style.marginLeft = '-280px';
        img2.style.marginTop = '260px';
        img2.style.borderRadius = '0px';
        img2.style.boxShadow = '0 2px 10px rgb(36, 250, 3)';
        img2.style.display = 'block';
        img2.alt = 'Histogram 2';
        img2.width = 260;
        container3.appendChild(img2);

        const container4 = document.getElementById('img4');
        container4.innerHTML = '';
        const img4 = document.createElement('img');
        img4.src = data.histogram3;
        img4.style.position = 'absolute';
        img4.style.backgroundColor = 'rgba(255, 255, 255, 0.53)';
        img4.style.padding = '10px';
        img4.style.marginLeft = '-280px';
        img4.style.marginTop = '500px';
        img4.style.borderRadius = '0px';
        img4.style.boxShadow = '0 2px 10px rgb(36, 250, 3))';
        img4.style.display = 'block';
        img4.alt = 'Histogram 2';
        img4.width = 260;
        container4.appendChild(img4);

        const container5 = document.getElementById('img5');
        container5.innerHTML = '';
        const img5 = document.createElement('img');
        img5.src = data.confidence_graph;
        img5.style.position = 'absolute';
        img5.style.backgroundColor = 'rgba(255, 255, 255, 0.53)';
        img5.style.padding = '10px';
        img5.style.marginLeft = '10px';
        img5.style.marginTop = '20px';
        img5.style.borderRadius = '0px';
        img5.style.boxShadow = '0 2px 10px rgb(36, 250, 3)';
        img5.style.display = 'block';
        img5.alt = 'Histogram 2';
        img5.width = 260;
        container3.appendChild(img5);

        const container6 = document.getElementById('img6');
        container6.innerHTML = '';
        const img6 = document.createElement('img');
        img6.src = data.handuh_graph;
        img6.style.position = 'absolute';
        img6.style.backgroundColor = 'rgba(255, 255, 255, 0.53)';
        img6.style.padding = '10px';
        img6.style.marginLeft = '10px';
        img6.style.marginTop = '260px';
        img6.style.borderRadius = '0px';
        img6.style.boxShadow = '0 2px 10px rgb(36, 250, 3))';
        img6.style.display = 'block';
        img6.alt = 'Histogram 2';
        img6.width = 260;
        
        container3.appendChild(img6);

        const container7 = document.getElementById('img7');
        container7.innerHTML = '';
        const img7 = document.createElement('img');
        img7.src = data.confusion_matrix;
        img7.style.position = 'absolute';
        img7.style.backgroundColor = 'rgba(255, 255, 255, 0.53)';
        img7.style.padding = '10px';
        img7.style.marginLeft = '10px';
        img7.style.marginTop = '500px';
        img7.style.borderRadius = '0px';
        img7.style.boxShadow = '0 2px 10px rgb(36, 250, 3)';
        img7.style.display = 'block'; 
        img7.width = 260;
        container3.appendChild(img7);
        container5.appendChild(img5);
        container6.appendChild(img6);
        container7.appendChild(img7);
    
        document.getElementById('result').style.display = 'block';
    })
    .catch(error => console.error('Error:', error));
}
