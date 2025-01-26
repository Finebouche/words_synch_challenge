import express from 'express';
import { Game } from '../database.js'; // Adjust the path to match your project structure
import { Op } from 'sequelize';

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

// POST /game/number_games
router.post('/number_games', async (req, res) => {
  try {
    const { playerId } = req.body;

    // 1) Count the number of games played by the player against a bot (where botId is not null) and where status is "won" or "lost"
    const gamesPlayedAgainstBot = await Game.count({ where: {
        player1Id: playerId,
        botId: { [Op.ne]: null },
        status: { [Op.in]: ['won', 'lost'] } }
    });

    // 2) Count the number of games played by the player against another human (where botId is null) and where status is "won" or "lost"
    const gamesPlayedAgainstHuman = await Game.count({ where: {
        player1Id: playerId,
        botId: null,
        status: { [Op.in]: ['won', 'lost'] } }
    });

    return res.json({ success: true, gamesPlayedAgainstBot, gamesPlayedAgainstHuman });
  } catch (err) {
    console.error('Error counting games played:', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
});


export default router;