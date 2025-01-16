import { Router } from 'express';
import path from 'path';

const router = Router();

// Middleware for token authentication
const authenticateDownload = (req, res, next) => {
  const token = req.headers['x-download-token'];
  const DOWNLOAD_TOKEN = process.env.DOWNLOAD_TOKEN;

  if (token && token === DOWNLOAD_TOKEN) {
    next(); // Token matches, proceed to download
  } else {
    res.status(403).send('Access denied. Invalid or missing token.');
  }
};

// Route to download the database file
router.get('/download-database', authenticateDownload, (req, res) => {
  res.download(path.resolve('word_sync.db'), 'word_sync.db', (err) => {
    if (err) {
      console.error('Error sending the database file:', err);
      res.status(500).send('Error downloading the database');
    }
  });
});

export default router;