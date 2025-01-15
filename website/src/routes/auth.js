import express from 'express';
import { Player } from '../database.js';

const router = express.Router();

// USER AUTHENTICATION

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


export default router;