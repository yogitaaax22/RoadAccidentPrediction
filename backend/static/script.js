let map = L.map('map').setView([20.5937, 78.9629], 5)

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{

maxZoom:19

}).addTo(map)

let marker


map.on("click", function(e){

let lat = e.latlng.lat
let lon = e.latlng.lng

document.getElementById("lat").value = lat
document.getElementById("lon").value = lon

if(marker){
map.removeLayer(marker)
}

marker = L.marker([lat,lon]).addTo(map)

})


function searchLocation(){

let location = document.getElementById("location").value

fetch("https://nominatim.openstreetmap.org/search?format=json&q="+location)

.then(res=>res.json())

.then(data=>{

let lat = data[0].lat
let lon = data[0].lon

document.getElementById("lat").value = lat
document.getElementById("lon").value = lon


map.setView([lat,lon],12)

if(marker){
map.removeLayer(marker)
}

marker = L.marker([lat,lon]).addTo(map)

})

}


function predictRisk() {
    console.log("Predict button clicked!"); // <--- ADD THIS
    let lat = document.getElementById("lat").value;
    let lon = document.getElementById("lon").value;

    if(!lat || !lon) {
        alert("Please select a location first!");
        return;
    }

    document.getElementById("risk").innerHTML = "Predicting...";

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            latitude: parseFloat(lat),
            longitude: parseFloat(lon)
        })
    })
    .then(res => {
        console.log("Response status:", res.status); // <--- ADD THIS
        return res.json();
    })
    .then(data => {
        console.log("Data received:", data); // <--- ADD THIS
        // Use 'risk_level' to match your Python JSON key
        document.getElementById("risk").innerHTML = data.risk_level || data.risk || "Unknown";
    })
    .catch(err => {
        console.error("Fetch Error:", err);
        document.getElementById("risk").innerHTML = "Error: Check Console";
    });
}
    
