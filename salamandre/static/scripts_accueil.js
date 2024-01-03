
var bouton = document.getElementById("goindex");
bouton.addEventListener("click",function(){
    window.location.href = "../../index.html";
});

var bouton2 = document.getElementById("godonnee");
bouton2.addEventListener("click",function(){
    window.location.href = "../../donnee.html";
});

var lat = 0;
var long =0;

axios.get('http://127.0.0.1:5000/getlast')
    .then (function (response){
    console.log(response.data);
    var data = response.data;
    console.log(data);
    console.log(data["latitude"]);
    lat = data['latitude'];
    long = data['longitude'];
    })
    .catch(function(error){
        console.log(error);
    });

// Based on this video https://dev.to/thedevdrawer/geolocation-tutorial-get-user-location-using-vanilla-js-46a on 25/10/2022
    class Geolocation {


        showPosition() {
            console.log('Bouton cliqué');
            console.log("latitude: ", lat, 'longitude: ', long);
            if (lat !==0 && long !==0){
                let mapContainer = document.querySelector("#map")
            mapContainer.style.display = "block"

            const map = L.map("map").setView([lat, long], 13)
            L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
                maxZoom: 18,
                attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);

            console.log(map);

            const marker = L.marker([lat, long]).addTo(map)
            } else {
                alert("Les données GPS ne sont pas présentes dans le header de la photo")
            }
        }
    }

const showPosition = document.querySelector("#showPosition")
showPosition.addEventListener("click", function (e) {
    e.preventDefault()
    new Geolocation().showPosition()
})

