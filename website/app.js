const express = require('express');
const path = require("path");
const app = express()

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "index.html"));
});


app.use(express.json());

// Array of random words
const randomWords = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew'];

// Endpoint to get a random word
app.get('/getRandomWord', (req, res) => {
   //  to replace with LLM request
    const randomIndex = Math.floor(Math.random() * randomWords.length);
    const randomWord = randomWords[randomIndex];
    res.json({ randomWord: randomWord });
});

module.exports = app