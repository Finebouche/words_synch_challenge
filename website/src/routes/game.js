// In your server code (e.g., app.js or a specific route file)
import express from 'express';
import { Game } from '../database.js'; // Adjust the path to match your project structure

const router = express.Router();

// POST /game/answers
router.post('/answers', async (req, res) => {
  try {
    const { gameId, strategyUsed, otherPlayerStrategy } = req.body;

    // 1) Find the game in the database
    const dbGame = await Game.findOne({ where: { gameId } });
    if (!dbGame) {
      return res.status(404).json({ success: false, message: 'Game not found.' });
    }

    let existingAnswers = dbGame.surveyAnswers ? JSON.parse(dbGame.surveyAnswers) : [];

    existingAnswers.push({
      timestamp: new Date().toISOString(),
      strategyUsed,
      otherPlayerStrategy
    });

    dbGame.surveyAnswers = JSON.stringify(existingAnswers);
    await dbGame.save();

    // 3) Return success
    return res.json({ success: true });
  } catch (err) {
    console.error('Error saving questionnaire answers:', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
});

export default router;