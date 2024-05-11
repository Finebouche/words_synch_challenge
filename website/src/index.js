const app = require('./app');
const port = process.env.PORT || 4000;
const { initDb } = require('./database');

// Somewhere in your server setup
initDb().catch(err => {
  console.error('Failed to initialize database:', err);
  process.exit(1); // Exit if database initialization fails
});


app.listen(port, () => {
    console.log(`Listening on port ${port}`);
});