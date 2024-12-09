import app from './app.js';
import { initDb } from './database.js';

const port = process.env.PORT || 4000;

// Initialize the database before starting the server
initDb().then(() => {
      app.listen(port, () => {
        console.log(`Listening on port ${port}`);
    });
}).catch(err => {
  console.error('Failed to initialize database:', err);
  process.exit(1); // Exit if database initialization fails
});
