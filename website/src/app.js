import express from 'express';
import path, { dirname } from 'path';
import { fileURLToPath } from 'url';
import authRoutes from './routes/auth.js';
import modelRoutes from './routes/model_game.js';
import databaseRoutes from './routes/database_sync.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

// Serve all static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.use(express.json());

app.use('/auth', authRoutes);
app.use('/model', modelRoutes);
app.use('/database', databaseRoutes);
// NOTE: no need to do `app.use('/players', playersRoutes)`
// because the "players" logic is Socket.IO-based, not typical route-based.

export default app;