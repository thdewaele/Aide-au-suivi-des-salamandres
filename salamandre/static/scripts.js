// Based on : https://medium.com/front-end-weekly/draw-an-image-in-canvas-using-javascript-%EF%B8%8F-2f75b7232c63 consulté le 20/02/23

var imagedisplay = 0;


function makePostRequest(path, queryObj){
    axios.post(path,queryObj).then(
        (response)=>{
            var result = response.data;
            console.log(result);
        },
        (error)=>{
            console.log(error);
        }
    );

}
document.addEventListener('DOMContentLoaded',(ev)=> {

    let input = document.getElementById('capture');

    input.addEventListener('change', async (ev) => {
        if (ev.target.files) {
            let pict = ev.target.files[0];
            var reader = new FileReader();
            reader.readAsDataURL(pict);
            var canvas = document.getElementById('canvas');
            canvas.width = 640;
            canvas.height = 900;
            var ctx = canvas.getContext('2d');

            reader.onload = function (e) {
                var image = new Image();
                image.src = e.target.result;
                image.onload = function (ev) {
                    ctx.drawImage(image, 0, 0, 640, 900);
                    imagedisplay = 1;
                    //console.log('display =1 ');
                }
            }
        }
    })
})

var lat = 0;
var long = 0;
BestRendering.InstallFileUploader('capture', 'http://127.0.0.1:5000/img', 'photo', function(response){
    var anwser = BestRendering.ParseJsonFromBackendUpload(response.data);
    lat = anwser['lat'];
    long = anwser['long'];
    console.log(anwser);
});


// Based on this video https://dev.to/thedevdrawer/geolocation-tutorial-get-user-location-using-vanilla-js-46a on 25/10/2022
    class Geolocation {


        showPosition() {
            if (lat !==0 && long !==0){
                let mapContainer = document.querySelector("#map")
            mapContainer.style.display = "block"

            const map = L.map("map").setView([lat, long], 13)
            const tiles = L.tileLayer(
                "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                {
                    maxZoom: 19,
                    attribution:
                        '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                }
            ).addTo(map)

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
var a = 0;
var b = 0;
var points = [];
points.push([a, b]);
var points2 =[];
points2.push([a, b]);
var test = 0;
var dist = 0;
var taillepiece =0;
var type = 0;

const btn = document.querySelector('#btn');
const sb = document.querySelector('#piece')
btn.onclick =(event)=>{
    event.preventDefault();
    type = sb.value;
    console.log("Type de pièce:", type);
}



const drawsal = document.querySelector("#drawsal")

drawsal.addEventListener("click", function(e){
    var i =1;
    if (test==0 || test==1) {
        document.addEventListener('click', function (event) {

            var c = document.getElementById("canvas");
            var ctx = c.getContext('2d');
            const rect = c.getBoundingClientRect()
            var x = event.clientX - rect.left;
            var y = event.clientY - rect.top;

            points.push([x, y]);
            if (test == 0) {
                points.splice(0, 2);
                test = 1;
            }

            if (points.length > 1 && test==1) {
                ctx.moveTo(points[i - 1][0], points[i - 1][1]);
                ctx.lineTo(points[i][0], points[i][1]);
                ctx.lineWidth = 5;
                ctx.strokeStyle = '#ff0000';
                ctx.stroke();
                dist = Number(dist + Math.sqrt(Math.pow(points[i][0] - points[i - 1][0], 2) + Math.pow([points[i][1] - points[i - 1][1]], 2)));
                i = i + 1;

            }
            //console.log([x, y]);
            //console.log('dist: ', dist)
        })
    }
})

const drawfin = document.querySelector("#drawfin")
var taille_pixel_sal =0;
drawfin.addEventListener("click",function (e){
    test = 2;
    taille_pixel_sal = dist;
    console.log('taille en pixel sal', taille_pixel_sal);
})

const drawpiece = document.querySelector("#drawpiece")

drawpiece.addEventListener("click", function(e){
    var i =1;
    if (test==2) {
        document.addEventListener('click', function (event) {

            var c = document.getElementById("canvas");
            var ctx = c.getContext('2d');
            const rect = c.getBoundingClientRect()
            var x = event.clientX - rect.left;
            var y = event.clientY - rect.top;

            points2.push([x, y]);
            if (test == 2) {
                points2.splice(0, 2);
                test = 3;
            }

            if (points2.length > 1 && test==3) {
                ctx.moveTo(points2[i - 1][0], points2[i - 1][1]);
                ctx.lineTo(points2[i][0], points2[i][1]);
                ctx.lineWidth = 5;
                ctx.stroke();
                taillepiece = Number(taillepiece + Math.sqrt(Math.pow(points2[i][0] - points2[i - 1][0], 2) + Math.pow([points2[i][1] - points2[i - 1][1]], 2)));
                i = i + 1;

            }
            //console.log([x, y]);
            //console.log('taille: ', taillepiece)
        })
    }
})

const drawpiecefin = document.querySelector("#drawpiecefin")
drawpiecefin.addEventListener("click",function (e){
    test = 4;
    const taille_pixel_piece = taillepiece;
    var piece;
    if (type == 0){
        piece = 2.325;
    }else if (type == 1){
        piece = 2.575;
    }else if (type ==2){
        piece = 2.425;
    }else{
        piece = 1;
    }
    //console.log('taille en pixel sal', taille_pixel_sal);
    if (dist !==0){
        var pixel_cm = taille_pixel_piece/piece;
        console.log("1 cm = pixel ", pixel_cm);
        var taille_salamandre = taille_pixel_sal/pixel_cm;
        var taille_cm = taille_salamandre.toPrecision(5);
        console.log("Taille salamandre en cm: ", taille_salamandre);
        document.getElementById("taille").innerHTML = "Estimation de la taille de la salamandre en cm: " + taille_cm;
    }
})