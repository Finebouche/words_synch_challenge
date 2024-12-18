import express from 'express';
import { Player } from '../database.js';

const router = express.Router();

// USER AUTHENTICATION

router.post('/login', async (req, res) => {
    const { playerId } = req.body;
    try {
        const player = await Player.findByPk(playerId);
        if (player) {
            res.json({ pseudonym: player.pseudonym, playerId: player.playerId });
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

router.post('/update-pseudonym', async (req, res) => {
    const { playerId, pseudonym } = req.body;
    if (!pseudonym || pseudonym.length < 3) {
        return res.status(400).send('Pseudonym must be at least 3 characters long');
    } else {
        try {
            const player = await Player.findByPk(playerId);
            if (player) {
                player.pseudonym = pseudonym;
                await player.save();
                res.send('Pseudonym updated successfully');
            } else {
                res.status(404).send('User not found');
            }
        } catch (error) {
            res.status(500).send('Server error');
        }
    }
});

export default router;