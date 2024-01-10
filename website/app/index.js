import http from 'http';
var app = require('./config').default();

// Serve static files from public directory
app.use(express.static('/public'));
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

// Start the server
http.createServer(app).listen(app.get('port'), () => {
    console.log("Express Server Runing on port"+ app.get('port'));
});

export default app;