let map = L.map('map').setView([20.5937, 78.9629], 5)

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
maxZoom:19
}).addTo(map)

let marker

/* Map Click to Select Location */

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


/* SEARCH LOCATION BUTTON (YOUR ORIGINAL FUNCTION) */

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


/* AUTOCOMPLETE SUGGESTIONS (NEW FEATURE) */

let locationInput = document.getElementById("location")
let suggestionBox = document.getElementById("suggestions")

locationInput.addEventListener("input", function(){

let query = this.value

if(query.length < 3){
suggestionBox.innerHTML = ""
return
}

fetch("https://nominatim.openstreetmap.org/search?format=json&q="+query)

.then(res=>res.json())

.then(data=>{

suggestionBox.innerHTML=""

data.forEach(place=>{

let li = document.createElement("li")

li.textContent = place.display_name

li.onclick = function(){

document.getElementById("location").value = place.display_name
document.getElementById("lat").value = place.lat
document.getElementById("lon").value = place.lon

map.setView([place.lat,place.lon],12)

if(marker){
map.removeLayer(marker)
}

marker = L.marker([place.lat,place.lon]).addTo(map)

suggestionBox.innerHTML=""

}

suggestionBox.appendChild(li)

})

})

})


/* WEATHER FUNCTION (NEW FEATURE) */

function getWeather(){

let lat = document.getElementById("lat").value
let lon = document.getElementById("lon").value

let apiKey = "YOUR_OPENWEATHER_API_KEY"

fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`)

.then(res=>res.json())

.then(data=>{

document.getElementById("weather").innerHTML =
"Temperature: " + data.main.temp + "°C , Condition: " + data.weather[0].main

})

}


/* ML MODEL PREDICTION (YOUR ORIGINAL BACKEND CALL) */

function predictRisk(){

let lat = document.getElementById("lat").value
let lon = document.getElementById("lon").value

document.getElementById("risk").innerHTML = "Predicting..."

fetch("/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({
latitude: parseFloat(lat),
longitude: parseFloat(lon)
})

})

.then(res=>res.json())

.then(data=>{

document.getElementById("risk").innerHTML = data.risk
document.getElementById("solution").innerHTML = data.solution

})

}
