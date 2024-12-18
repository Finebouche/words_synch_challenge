import express from 'express';
import path, { dirname } from 'path';
import { fileURLToPath } from 'url';
import authRoutes from './routes/auth.js';
import modelRoutes from './routes/model.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

// Serve all static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.use(express.json());

// Mount the auth routes under /auth
app.use('/auth', authRoutes);

// Mount the model routes under /model
app.use('/model', modelRoutes);

export default app;