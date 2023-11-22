/**
function updateTable(){
    axios.get('/getdata')
        .then(function(response){
            const data = response.data;
            const tableBody = document.querySelector('salamandres-table');
            tableBody.innerHTML='';
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
        console.(error);
    });
}
document.addEventListener('DOMContentLoaded', function (){
    updateTable();
});
setInterval(updateTable, 5000);
**/
