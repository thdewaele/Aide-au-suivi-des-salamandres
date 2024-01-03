// Récupération de toutes les lignes du tableau sauf l'en-tête
var tableRows = Array.from(document.querySelectorAll('#salamandres-table tbody tr'));

// Trier les lignes du tableau en fonction de la colonne ID
tableRows.sort(function(a, b) {
    var idA = parseInt(a.cells[0].textContent);
    var idB = parseInt(b.cells[0].textContent);
    return idA - idB;
});

// Supprimer les lignes du tableau existant
var tableBody = document.querySelector('#salamandres-table tbody');
tableBody.innerHTML = '';

// Réinsérer les lignes triées dans le tableau
tableRows.forEach(function(row) {
    tableBody.appendChild(row);
});
/*
function updateTable(){

    axios.get('/getdata')
        .then(function(response){
            const data = response.data;
            const tableBody = document.querySelector('salamandres-table');
            data.forEach(function(salamandre){
                const row = tableBody.insertRow();
                const cell1 = row.insertCell(0);
                const cell2 = row.insertCell(1);
                const cell3 = row.insertCell(2);
                const cell4 = row.insertCell(3);
                const cell5 = row.insertCell(4);
                const cell6 = row.insertCell(5);
                const cell7 = row.insertCell(6);
                const cell8 = row.insertCell(7);

                cell1.textContent = salamandre.id;
                cell2.textContent = salamandre.filename;
                cell3.textContent = salamandre.file;
                cell4.textContent = salamandre.latitude;
                cell5.textContent = salamandre.longitude;
                cell6.textContent = salamandre.focal;
                cell7.textContent = salamandre.date;
                cell8.textContent = salamandre.size;
            });
        })
        .catch(function (error){
        console.error(error);
    });
}
document.addEventListener('DOMContentLoaded', function (){
    updateTable();
});
setInterval(updateTable, 5000);
 */
