const BASE_URL = "https://krishixpert-ai-based-crop-disease.onrender.com";

async function predictDisease() 
{
  try {

    const predictBtn = document.getElementById("predictBtn");
    if (predictBtn) {
      predictBtn.disabled = true;
      predictBtn.style.opacity = 0.6;
    }

    const resp = await fetch(`${BASE_URL}/predict?t=${Date.now()}`, {
      method: 'POST',
      cache: 'no-store',
      headers: {
        'Cache-Control': 'no-store, no-cache, must-revalidate',
        'Pragma': 'no-cache'
      }
    });

    if (!resp.ok) {
      console.error('Server returned', resp.status);
      resultBox.innerHTML = "❌ Prediction failed (server error).";
      if (predictBtn) { predictBtn.disabled = false; predictBtn.style.opacity = 1; }
      return;
    }

    const data = await resp.json();
    if (data.error) {
      console.error('Server error:', data);
      resultBox.innerHTML = `❌ ${data.error}`;
      if (predictBtn) { predictBtn.disabled = false; predictBtn.style.opacity = 1; }
      return;
    }

    
const ratio  = data.npk_ratio || {};
const pumps  = data.pump_times_ms || {};
const status = (data.esp32_pump_status === null || data.esp32_pump_status === undefined)
  ? 'No response'
  : data.esp32_pump_status;
        // Prediction details ( <p>Remedies: ${data.remedies}</p>)
        document.getElementById('predictionResult').innerHTML = `
            <div class="result-card">
                <h2>Prediction Result:</h2>
                <h3>Prediction: ${data.prediction}</h3>              
                <p>NPK Ratio: 1:2:4</p>
                <p>Pump N in ms:1000 </p> 
                <p>Pump N in ms: 2000</p> 
                <p>Pump N in ms: 4000</p> 
                <p>Pump total in ms: 14000 </p> 
            </div>
        `;


const main = document.getElementById("main-container");

Object.assign(main.style, {
    display: "flex",
    flexDirection: "column",
    gap: "30px",
    position: "relative",
    alignItems: "flex-start",
    marginLeft: "90px",
    marginTop: "30px"
});


    const container1 = document.getElementById('predictionResult1');
    container1.innerHTML = '';
    container1.style.width = "250px";
    container1.style.background = "rgba(255, 255, 255, 0.53)";
    container1.style.padding = "10px";
    container1.style.borderRadius = "10px";
    container1.innerHTML = `
    <p>Accuracy: ${data.confidence}%</p>
    <p>Loss: ${data.loss}%</p>
    <p>Healthy percentage: ${data.healthy_percentage}%</p>
    <p>Unhealthy percentage: ${data.unhealthy_percentage}%</p>
    <p>COUNT: ${data.count}</p>
    <P>Prediction Time: ${data.prediction_latency} ms</p> 
    <p>TIMETAMP: ${data.timestamp}</p>
    `;

    const container2= document.getElementById('count');
    container2.innerHTML = '';
    container2.style.width = "250px";
    container2.style.background = "rgba(255, 255, 255, 0.53)";
    container2.style.padding = "10px";
    container2.style.borderRadius = "10px";
    container2.innerHTML =`
        <h1> LIVE MONITOR</h1>
        <p>Temperature: <span id="temperature">27°C</span></p>
        <p>Humidity: <span id="humidity">38%</span></p>
        <p>Ph level: <span id="ph">6.2%</span></p>
        <p>Ligth Intensity: <span id="ligth">223.5Lux</span></p>
        `;

        
    // Processed Leaf Images
        
    const container = document.getElementById('image-container');

    container.innerHTML = '';
    container.style.display = 'grid';
    container.style.gridTemplateRows = 'repeat(3, auto)';
    container.style.gridAutoFlow = 'column'; 
    container.style.gap = '25px';
    container.style.justifyItems = 'center';

    // Image sources (order matters)
    const images = [
        data.image_black_background, 
        data.image_green_white,      
        data.image_white_red,        
        data.histogram1,            
        data.histogram2,             
        data.histogram3,
        data.confidence_graph ,
        data.handuh_graph,
        data.confusion_matrix,
        data.roc_curve
        ];

    // Create and append images
    images.forEach((src, index) => {
        const img = document.createElement('img');
        img.src = src;
        img.alt = `Image ${index + 1}`;

    // Image styling

        if (index < 3) 
        {
        img.style.width = '250px';   
        } 
        else  
            {
                img.style.width = '340px';   
             }
        img.style.padding = '10px';
        img.style.backgroundColor = 'rgba(255, 255, 255, 0.53)';
        img.style.display = 'block';
        container.appendChild(img);
    });
    
        document.getElementById('result').style.display = 'block';
    }

catch (error) {
    console.error('Error:', error);
    const resultBox = document.getElementById("resultBox");
    resultBox.innerHTML = "❌ Error connecting to server.";
  } finally {
    const predictBtn = document.getElementById("predictBtn");
    if (predictBtn) { predictBtn.disabled = false; predictBtn.style.opacity = 1; }
  }
}

    const toggleButton = document.getElementById("toggleSidebar");
    const sidebar = document.querySelector(".sidebar");
    const menuItems = document.querySelectorAll(".menu li");

    // Sidebar Toggle Function
    toggleButton.addEventListener("click", function () {
        sidebar.classList.toggle("collapsed");

        // Move button position based on sidebar state
        if (sidebar.classList.contains("collapsed")) {
            toggleButton.style.left = "90px";
        } else {
            toggleButton.style.left = "280px";
        }
    });

const leafBg=document.querySelector(".leaf-bg");
const leaves=["🍃","🌿","🌱","🍁","🍂"];

for(let i=0;i<20;i++){
  const leaf=document.createElement("div");
  leaf.className="leaf";
  leaf.innerText=leaves[Math.floor(Math.random()*5)];

  leaf.style.left=Math.random()*100+"vw";
  leaf.style.animationDuration=(6+Math.random()*8)+"s";
  leaf.style.fontSize=(18+Math.random()*18)+"px";

  leafBg.appendChild(leaf);
}


const SERVER_URL = "http://127.0.0.1:5000/health";
const indicator = document.getElementById("server-indicator");
const latencyText = document.getElementById("server-latency");

async function checkServer() {
    const startTime = performance.now();

    try {
        const res = await fetch(SERVER_URL, { cache: "no-store" });
        const endTime = performance.now();

        if (res.ok) {
            const latency = Math.round(endTime - startTime);
            indicator.classList.add("online");
            latencyText.textContent = `${latency} ms`;
        } else {
            indicator.classList.remove("online");
            latencyText.textContent = "";
        }
    } catch (err) {
        indicator.classList.remove("online");
        latencyText.textContent = "";
    }
}

setInterval(checkServer, 3000);
checkServer();


const STATS_URL = "http://127.0.0.1:5000/system_stats";

async function updateSystemStats() {
    try {
        const res = await fetch(STATS_URL, { cache: "no-store" });
        const data = await res.json();

        document.getElementById("system-monitor").innerText =
            `    System Usage
                 CPU: ${data.cpu_percent}%
                 RAM: ${data.ram_used_mb} / ${data.ram_total_mb} MB
                 RAM Usage: ${data.ram_percent}%
            `;

const main = document.getElementById("system-monitor");

Object.assign(main.style, {
    background: "rgba(255,255,255,0.6)",
    position: "absolute",
    padding: "10px",
    top: "60px",
    left: "900px",
    borderRadius: "8px",
    fontSize: "13px",
    width: "max-content"
});

main.innerHTML = `
    <strong>System Usage</strong><br>
    CPU: ${data.cpu_percent}%<br>
    RAM: ${data.ram_used_mb} / ${data.ram_total_mb} MB<br>
    RAM Usage: ${data.ram_percent}%
`;


    } catch (err) {
        console.error("System stats error", err);
    }
}

setInterval(updateSystemStats, 2000);
updateSystemStats();    
