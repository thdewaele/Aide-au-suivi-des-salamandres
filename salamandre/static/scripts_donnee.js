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

