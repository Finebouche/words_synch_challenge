const express = require('express');
const app = express();
var PORT = process.env.PORT || 3000;

// Serve static files from a specific directory (e.g., 'public')
app.use(express.static('../frontend'));

// Array of random words
const randomWords = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew'];

app.use(express.json());

// Endpoint to get a random word
app.get('/getRandomWord', (req, res) => {
   //  to replace with LLM request
    const randomIndex = Math.floor(Math.random() * randomWords.length);
    const randomWord = randomWords[randomIndex];
    res.json({ randomWord: randomWord });
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});

module.exports = app;