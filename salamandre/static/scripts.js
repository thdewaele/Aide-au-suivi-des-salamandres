
// Based on : https://medium.com/front-end-weekly/draw-an-image-in-canvas-using-javascript-%EF%B8%8F-2f75b7232c63 consulté le 20/02/23

var imagedisplay = 0;
var test = 0;

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

var name = null;
document.addEventListener('DOMContentLoaded',(ev)=> {

    let input = document.getElementById('capture');

    input.addEventListener('change', async (ev) => {
        if (ev.target.files) {
            let pict = ev.target.files[0];
            name = pict;
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





BestRendering.InstallFileUploader('capture', 'http://127.0.0.1:5000/addpict', 'photo', function (response) {
    var anwser = BestRendering.ParseJsonFromBackendUpload(response.data);
    console.log(anwser);
});





// Based on this video https://dev.to/thedevdrawer/geolocation-tutorial-get-user-location-using-vanilla-js-46a on 25/10/2022
    class Geolocation {


        showPosition(lat, long) {
            console.log(lat, long)
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
    axios.get('http://127.0.0.1:5000/getlast')
    .then (function (response){
    console.log(response.data);
    var data = response.data;
    console.log(data);
    console.log(data["latitude"]);
    var lat = data['latitude'];
    var long = data['longitude'];
    new Geolocation().showPosition(lat,long)
    })
    .catch(function(error){
        console.log(error);
    });

})

var a = 0;
var b = 0;
var points = [];
points.push([a, b]);
var points2 =[];
points2.push([a, b]);

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
    document.getElementById("taille").innerHTML = "Identification de la salamandre terminée";
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

const drawpiecefin = document.querySelector("#drawpiecefin");
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
    if (dist !==0) {
        var pixel_cm = taille_pixel_piece / piece;
        console.log("1 cm = pixel ", pixel_cm);
        var taille_salamandre = taille_pixel_sal / pixel_cm;
        var taille_cm = taille_salamandre.toPrecision(5);
        console.log("Taille salamandre en cm: ", taille_salamandre);
        let inputElement = document.getElementById('capture');
        var filename = null;
        if (inputElement.files.length>0){
            filename = inputElement.files[0].name;
            console.log(filename);
        }
        document.getElementById("taille").innerHTML = "Estimation de la taille de la salamandre en cm: " + taille_cm;
        document.getElementById("ajoutdb").innerHTML ="Votre photo a bien été ajouté à la base de données";
        const taille = taille_cm;
        axios.post('http://127.0.0.1:5000/addtaille', {
            size: taille,
            filename: filename
        })
        .then((response) => {
            console.log(response);
        });

    }
});

const tabcomplet= document.querySelector("#tableaumandel");
tabcomplet.addEventListener("click",function (e) {
     document.getElementById("fin_encodage").innerHTML = "Encodage ajouté à la db";
    var tableau = document.getElementById("tableau");
    var pourc = document.getElementById("pourcinput").value;
    console.log(pourc);
    if (pourc != NaN){
        console.log(pourc);
        pourc = parseInt(pourc);
    }else{
        pourc = 95;
    }
    var lignes = tableau.getElementsByTagName("tr");

    var tab = [];


    for (var i = 0; i < lignes.length; i++) {
        var cells = lignes[i].getElementsByTagName("td");
        var ligne = [];

        for (var j = 0; j < cells.length; j++) {
            var elem = cells[j];
            if (elem != null){
                if (elem.querySelector("input") != null) {
                    var valeur = elem.querySelector("input").value;

                }
                else{
                    var valeur = 0;
                }
            }else{
                var valeur = 0;
            }


            ligne.push(valeur);
        }
        tab.push(ligne)
    }




    var indice = -1;
    axios.post('http://127.0.0.1:5000/identification',{
        tableau: tab, pourc : pourc
    })
        .then(function (response) {
            console.log(response.data);
            var data = response.data;
            lat = data['latitude'];
            long = data['longitude'];
            date = data['date'];
            indice = data['index'];
            pourcentage = data['pourcentage'];
            if (lat != 0 && long != 0) {
                document.getElementById("salsimilaire").innerHTML = "Une salamandre similaire à " + pourcentage + "% a été observée le " + date + "à cette position: latitude: " + lat + ", longitude: " + long;
            }
            console.log(indice);
            if (indice > 0){
                 return axios.get('http://127.0.0.1:5000/get_image',{
                   params:{ index: indice}
                })
                    .then(response => {
                        const buffer= response.data;
                        if (buffer != null){
                            console.log(buffer)
                            const binary = new Uint8Array(buffer);
                            console.log(binary)
                            const imageData = buffer.reduce((data, byte)=>data+String.fromCharCode(byte),'');
                            console.log(imageData);
                            const base64Image= btoa(imageData)
                            const imgElement = document.createElement('img')
                            //const imageData = Buffer.from(response.data, 'binary').toString('base64');
                            //const imgElement = document.createElement('img');
                            imgElement.src = `data:image/jpeg;ascii,${base64Image}`;
                            document.getElementById('image-container').appendChild(imgElement);
                            console.log (response.data);
                        }
                    }).catch(error=>{
                        console.error('Erreur lors de la récupération de l\' image: ', error);
                })
            }

    })
    .catch(function (error){
        console.error(error)
    });



    console.log(tab);


});




