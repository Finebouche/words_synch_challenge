import express from 'express';
import { Game } from '../database.js'; // Adjust the path to match your project structure
import { Op } from 'sequelize';

const router = express.Router();

// POST /game/answers
router.post('/answers', async (req, res) => {
  try {
    const {
      gameId,
      playerId,  // Now we check which player is submitting the data
      quantitativeStrategyUsed,
      qualitativeStrategyUsed,
      quantitativeOtherPlayerStrategy,
      qualitativeOtherPlayerStrategy,
      otherPlayerUnderstoodYourStrategies,
      didYouUnderstandOtherPlayerStrategy,
      otherPlayerRating,
      connectionFeeling
    } = req.body;

    // 1) Find the game in the database
    const dbGame = await Game.findOne({ where: { gameId } });
    if (!dbGame) {
      return res.status(404).json({ success: false, message: 'Game not found.' });
    }

    // 2) Determine if player is Player 1 or Player 2
    let isPlayer1 = dbGame.player1Id === playerId;
    let isPlayer2 = dbGame.player2Id === playerId;

    if (!isPlayer1 && !isPlayer2) {
      return res.status(403).json({ success: false, message: 'Player not part of this game.' });
    }

    // 3) Select the correct field (surveyAnswers1 or surveyAnswers2)
    let answerField = isPlayer1 ? 'surveyAnswers1' : 'surveyAnswers2';

    let existingAnswers = [];

    // 4) Append new answers
    existingAnswers.push({
      timestamp: new Date().toISOString(),
      quantitativeStrategyUsed,
      qualitativeStrategyUsed,
      quantitativeOtherPlayerStrategy,
      qualitativeOtherPlayerStrategy,
      otherPlayerUnderstoodYourStrategies,
      didYouUnderstandOtherPlayerStrategy,
      otherPlayerRating,
      connectionFeeling
    });

    // 5) Save back to the correct field
    dbGame[answerField] = JSON.stringify(existingAnswers);
    await dbGame.save();


    // 6) Return success response
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

    // 1) Count the number of completed games played by the player against a bot (where botId is not null)
    const gamesPlayedAgainstBot = await Game.count({
      where: {
        player1Id: playerId,
        botId: { [Op.ne]: null },
        status: { [Op.in]: ['won', 'lost'] }
      }
    });

    // 2) Count the number of completed games played by the player against another human (where botId is null)
    const gamesPlayedAgainstHuman = await Game.count({
      where: {
          botId: null, // Ensure it's a human vs. human game
          status: { [Op.in]: ['won', 'lost'] }, // Only count completed games
          [Op.or]: [
              { player1Id: playerId }, // Player was Player 1
              { player2Id: playerId }  // Player was Player 2
          ]
      }
    });

    return res.json({ success: true, gamesPlayedAgainstBot, gamesPlayedAgainstHuman });
  } catch (err) {
    console.error('Error counting games played:', err);
    return res.status(500).json({ success: false, message: `Internal server error: ${err}` });
  }
});


export default router;