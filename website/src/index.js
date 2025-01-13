import http from 'http';
import app from './app.js';
import { initDb } from './database.js';
import initPlayersSocket from './routes/player_game.js';

const port = process.env.PORT || 4000;

// Initialize the database before starting the server
initDb()
  .then(() => {
    // 1) Create raw HTTP server
    const server = http.createServer(app);

    // 2) Attach Socket.IO logic
    initPlayersSocket(server);

    // 3) Now start listening
    server.listen(port, () => {
      console.log(`Listening on port ${port}`);
    });
  })
  .catch(err => {
    console.error('Failed to initialize database:', err);
    process.exit(1); // Exit if database initialization fails
  });