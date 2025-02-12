import express from 'express';
import { Player, Game } from '../database.js';
import { Op } from 'sequelize';

const router = express.Router();

router.post('/login', async (req, res) => {
    const { playerId } = req.body;
    try {
        const player = await Player.findByPk(playerId);
        if (player) {
            res.json({
            pseudonym: player.pseudonym,
            playerId: player.playerId,
            ageGroup: player.ageGroup,
            gender: player.gender,
            region: player.region,
            llmKnowledge: player.llmKnowledge
            });
        } else {
            res.status(404).send('User not found');
        }
    } catch (error) {
        res.status(500).send('Server error');
    }
});

router.post('/create', async (req, res) => {
    const { playerId } = req.body;
    try {
        const [player, created] = await Player.findOrCreate({
            where: { playerId: playerId },
            defaults: { playerId: playerId }
        });
        res.json({ pseudonym: player.pseudonym });
    } catch (error) {
        res.status(500).send('Server error');
    }
});

router.post('/update-profile', async (req, res) => {
  const { playerId, pseudonym, ageGroup, gender, region, llmKnowledge } = req.body;

  try {
    const player = await Player.findByPk(playerId);
    if (!player) {
      return res.status(404).send('User not found');
    }

    // Update fields if provided in the request body
    if (pseudonym !== undefined) player.pseudonym = pseudonym;
    if (ageGroup !== undefined) player.ageGroup = ageGroup;
    if (gender !== undefined) player.gender = gender;
    if (region !== undefined) player.region = region;
    if (llmKnowledge !== undefined) player.llmKnowledge = llmKnowledge;

    await player.save();
    return res.status(200).send('Profile updated successfully');
  } catch (error) {
    console.error(error);
    return res.status(500).send('Server error');
  }
});

// Endpoint to heck if player exists
router.post('/exists/', async (req, res) => {
    const { playerId } = req.body;

    try {
        const player = await Player.findByPk(playerId);
        if (player) {
            return res.json({ exists: true });
        } else {
            return res.json({ exists: false });
        }
    } catch (error) {
        console.error(error);
        return res.status(500).send('Server error');
    }
});

// Endpoint to know how many games a player has played for each combination of gameConfig
router.post('/games-config-count/', async (req, res) => {
    const { playerId } = req.body;

    try {
        // Get the gameConfigOrder of the player
        const player = await Player.findByPk(playerId);
        if (!player) {
            return res.status(404).send('User not found');
        }
        const gameConfigOrder = player.gameConfigOrder;

        // Count the number of games played for each gameConfig 'human_vs_human_(bot_shown)', 'human_vs_bot_(bot_shown)', 'human_vs_human_(human_shown)', 'human_vs_bot_(human_shown)'
        const gamesCount = {
            'human_vs_human_(bot_shown)': 0,
            'human_vs_bot_(bot_shown)': 0,
            'human_vs_human_(human_shown)': 0,
            'human_vs_bot_(human_shown)': 0
        }
        const games = await Game.findAll({
            where: {
                [Op.or]: [
                    { player1Id: playerId },
                    { player2Id: playerId }
                ]
            }
        });
        games.forEach(game => {
            const gameConfig = game.gameConfig;
            if (gamesCount[gameConfig]) {
                gamesCount[gameConfig]++;
            } else {
                gamesCount[gameConfig] = 1;
            }
        });

        // return the count of games played for each gameConfig and the gameConfigOrder of the player
        return res.json({ gamesCount, gameConfigOrder });
    } catch (error) {
            return res.status(500).send(`Server error : ${error}`);
    }
});

export default router;