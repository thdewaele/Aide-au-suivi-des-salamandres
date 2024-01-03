
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

var button_tab = document.getElementById("pretab");
button_tab.addEventListener("click", function (){
    axios.get('http://127.0.0.1:5000/getTable')
        .then(function(response){
            const data = response.data;
            const donnees= data["table"];
            //const tab = donnees[0]
            console.log(donnees)
            //console.log(tab)
            const tableauHTML = document.getElementById('tableau');
            console.log(donnees.length);
            console.log(donnees[0]);
            const tab = donnees;
            console.log(tab.length);
            console.log(tab[1].length);
            const element1 = tab[1][7];
            console.log(element1);
            const element2 = tab[1][8];
            const element3 = element2+element1;
            console.log(element3);
            const cellule1 = document.getElementById('cellule1');
            cellule1.querySelector('input[type="text"]').value= tab[0][6]+tab[0][7];
            const cellule2 = document.getElementById('cellule2');
            cellule2.querySelector('input[type="text"]').value= tab[1][5]+tab[1][4];
            const cellule3 = document.getElementById('cellule3');
            cellule3.querySelector('input[type="text"]').value= tab[1][8]+tab[1][9];
            const cellule4 = document.getElementById('cellule4');
            cellule4.querySelector('input[type="text"]').value= tab[1][6];
            const cellule5 = document.getElementById('cellule5');
            cellule5.querySelector('input[type="text"]').value= tab[1][7];
            const cellule6 = document.getElementById('cellule6');
            cellule6.querySelector('input[type="text"]').value= tab[2][5]+tab[2][4];
            const cellule7 = document.getElementById('cellule7');
            cellule7.querySelector('input[type="text"]').value= tab[2][6];
            const cellule8 = document.getElementById('cellule8');
            cellule8.querySelector('input[type="text"]').value= tab[2][7];
            const cellule9 = document.getElementById('cellule9');
            cellule9.querySelector('input[type="text"]').value= tab[2][8]+tab[2][9];
            const cellule10 = document.getElementById('cellule10');
            cellule10.querySelector('input[type="text"]').value= tab[3][6]+tab[3][5];
            const cellule11 = document.getElementById('cellule11');
            cellule11.querySelector('input[type="text"]').value= tab[3][7]+tab[3][8];
            const cellule12 = document.getElementById('cellule12');
            cellule12.querySelector('input[type="text"]').value= tab[4][2]+tab[4][1];
            const cellule13 = document.getElementById('cellule13');
            cellule13.querySelector('input[type="text"]').value= tab[4][3];
            const cellule14 = document.getElementById('cellule14');
            cellule14.querySelector('input[type="text"]').value= tab[4][4];
            const cellule15 = document.getElementById('cellule15');
            cellule15.querySelector('input[type="text"]').value= tab[4][8];
            const cellule16 = document.getElementById('cellule16');
            cellule16.querySelector('input[type="text"]').value= tab[4][9];
            const cellule17 = document.getElementById('cellule17');
            cellule17.querySelector('input[type="text"]').value= tab[4][10];
            const cellule18 = document.getElementById('cellule18');
            cellule18.querySelector('input[type="text"]').value= tab[4][5];
            const cellule19 = document.getElementById('cellule19');
            cellule19.querySelector('input[type="text"]').value= tab[4][6];
            const cellule20 = document.getElementById('cellule20');
            cellule20.querySelector('input[type="text"]').value= tab[4][7];
            const cellule21 = document.getElementById('cellule21');
            cellule21.querySelector('input[type="text"]').value= tab[5][5]+tab[5][4];
            const cellule22 = document.getElementById('cellule22');
            cellule22.querySelector('input[type="text"]').value= tab[5][6];
            const cellule23 = document.getElementById('cellule23');
            cellule23.querySelector('input[type="text"]').value= tab[5][7]+tab[5][8];
            const cellule24 = document.getElementById('cellule24');
            cellule24.querySelector('input[type="text"]').value= tab[6][5]+tab[6][4];
            const cellule25 = document.getElementById('cellule25');
            cellule25.querySelector('input[type="text"]').value= tab[6][6];
            const cellule26 = document.getElementById('cellule26');
            cellule26.querySelector('input[type="text"]').value= tab[6][7]+tab[6][8];
            const cellule27 = document.getElementById('cellule27');
            cellule27.querySelector('input[type="text"]').value= tab[7][5]+tab[7][4];
            const cellule28 = document.getElementById('cellule28');
            cellule28.querySelector('input[type="text"]').value= tab[7][6];
            const cellule29 = document.getElementById('cellule29');
            cellule29.querySelector('input[type="text"]').value= tab[7][7]+tab[7][8];
            const cellule30 = document.getElementById('cellule30');
            cellule30.querySelector('input[type="text"]').value= tab[8][5];
            const cellule31 = document.getElementById('cellule31');
            cellule31.querySelector('input[type="text"]').value= tab[8][6];
            const cellule32 = document.getElementById('cellule32');
            cellule32.querySelector('input[type="text"]').value= tab[8][7];
            const cellule33 = document.getElementById('cellule33');
            cellule33.querySelector('input[type="text"]').value= tab[8][1];
            const cellule34 = document.getElementById('cellule34');
            cellule34.querySelector('input[type="text"]').value= tab[8][2];
            const cellule35 = document.getElementById('cellule35');
            cellule35.querySelector('input[type="text"]').value= tab[8][3];
             const cellule36 = document.getElementById('cellule36');
            cellule36.querySelector('input[type="text"]').value= tab[8][4];
             const cellule37 = document.getElementById('cellule37');
            cellule37.querySelector('input[type="text"]').value= tab[8][8];
            const cellule38 = document.getElementById('cellule38');
            cellule38.querySelector('input[type="text"]').value= tab[8][9];
             const cellule39 = document.getElementById('cellule39');
            cellule39.querySelector('input[type="text"]').value= tab[8][10];
             const cellule40 = document.getElementById('cellule40');
            cellule40.querySelector('input[type="text"]').value= tab[8][1];
            const elem = tab[9][6]+tab[9][7]+ tab[10][6]+tab[10][7]+ tab[11][6]+tab[11][7]+ tab[12][6]+tab[12][7]+ tab[13][6]+tab[13][7];
            const cellule41 = document.getElementById('cellule41');
            cellule41.querySelector('input[type="text"]').value = elem;




        });
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
            if (lat >0 && long > 0) {
                document.getElementById("salsimilaire").innerHTML = "Une salamandre similaire à " + pourcentage + "% a été observée le " + date + "à cette position: latitude: " + lat + ", longitude: " + long;
            } else if (lat==0 && long==0){
                document.getElementById('salsimilaire').innerHTML='Une salamandre similaire à ' + pourcentage + '% a été obervée le ' + date + " mais les données de positions ne sont pas disponibles";
            }
            console.log(indice);
        })
        .catch(function (error){
            console.error(error)
        });
    document.getElementById("download").addEventListener("click",function (){
        return axios.get('http://127.0.0.1:5000/get_image', {
                    params: {index: indice},
                    responseType: 'blob'
                })
                    .then(function (response) {
                        console.log("Hello");

                        const blob = response.data;
                        if (blob instanceof Blob) {
                          const imageUrl = URL.createObjectURL(blob);
                          const image = new Image();

                          image.onload = function () {
                              const canvas = document.getElementById("target");
                              canvas.width = image.width;
                              canvas.height = image.height;
                              const ctx = canvas.getContext("2d");
                              ctx.drawImage(image, 0, 0);
                          };
                          image.src = imageUrl;

                          image.onerror = function () {
                              console.error("Erreur lors du chargement de l'image.", this.src);
                          };
                        }
                    })
                    .catch(error=>{
                        console.error('Erreur lors de la récupération de l\' image: ', error);
                     });
    });


    console.log(tab);


});
/*
                        const blob = new Blob([response.data], {
                            type: response.headers["content-type"]
                        });
                        //console.log(response.data);
                        console.log(blob);
                        var image = new Image();
                        var canvas = document.getElementById("target");
                        var ctx = canvas.getContext("2d");
                        console.log(ctx);
                        console.log(image);
                        image.src = URL.createObjectURL(blob);
                        console.log(image.src);
                        canvas.width = image.width;
                        canvas.height = image.height;
                        console.log(canvas.width);
                        console.log(canvas.height);
                        image.onload = function () {
                            //var canvas = document.getElementById("target");

                            ctx.drawImage(image,0,0);
                            console.log('On image.onload');
                            //var target = canvas.getContext("2d");
                             //target.drawImage(image, 0, 0);



                        };

                         */



