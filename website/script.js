document.getElementById('startGame').addEventListener('click', function() {
    document.getElementById('selections').style.display = 'none';
    document.getElementById('gameInput').style.display = 'block';
});

document.getElementById('submitWord').addEventListener('click', function() {
    var word = document.getElementById('gameWord').value.trim();
    if (word) {
        var messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.innerHTML = `${word} <span class="emoji">&#x1F60A;</span>`; // Face emoji
        document.getElementById('conversationArea').appendChild(messageDiv);
        document.getElementById('gameWord').value = ''; // Clear the input field
    }
});

document.getElementById('submitWord').addEventListener('click', function() {
    var player1Word = document.getElementById('player1').value.trim().toLowerCase();
    var player2Word = document.getElementById('player2').value.trim().toLowerCase();
    var status = document.getElementById('gameStatus');

    if (player1Word === player2Word) {
        status.textContent = 'Game Over: Both players chose ' + player1Word;
    } else {
        status.textContent = 'Keep playing...';
    }
});