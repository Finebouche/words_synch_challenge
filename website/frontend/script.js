document.getElementById('startGame').addEventListener('click', function() {
    document.getElementById('selections').style.display = 'none';
    document.getElementById('gameInput').style.display = 'block';
});

var past_words_array = []; // Array to store the words

document.getElementById('submitWord').addEventListener('click', function() {
    var word = document.getElementById('gameWord').value.trim();
    if (word) {
        past_words_array.push(word); // Add word to array

        var messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.innerHTML = `${word} <span class="emoji">&#x1F60A;</span>`; // Face emoji
        document.getElementById('conversationArea').appendChild(messageDiv);
        document.getElementById('gameWord').value = ''; // Clear the input field

        // Send request to Express.js backend
        fetch('/getRandomWord')
            .then(response => response.json())
            .then(data => {
                var randomWord = data.randomWord;
                past_words_array.push(randomWord); // Add received word to array

                // You can display this word or handle it as you wish
                // For example, appending it to the conversation area
                var randomWordDiv = document.createElement('div');
                randomWordDiv.classList.add('message');
                randomWordDiv.innerHTML = `${randomWord} <span class="emoji">&#x1F916;</span>`; // Modify as needed
                document.getElementById('conversationArea').appendChild(randomWordDiv);
            })
            .catch(error => {
                console.error('Error fetching random word:', error);
            });
    }
});
