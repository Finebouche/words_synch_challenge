import { Server } from 'socket.io';
import { Game, Player } from '../database.js';
import { v4 as uuidv4 } from 'uuid';

const MAX_NUMBER_OF_ROUNDS = 15;

export default function initPlayersSocket(server) {
  const io = new Server(server);

  // Keep a map of language -> waiting player
  const waitingPlayers = {};

  // In-memory store of running games
  const activeGames = {};

  // Helper function to return how many players are waiting in total
  function getWaitingPlayersCount() {
    return Object.keys(waitingPlayers).length;
    console.log('Waiting players:', waitingPlayers);
  }

  io.on('connection', (socket) => {
    console.log('A user connected:', socket.id);

    /**
     * 1) Player indicates their identity and language, then joins queue
     */
    socket.on('joinQueue', async ({ language, playerId }) => {
      console.log(
        `Socket ${socket.id} (playerId=${playerId}, language=${language}) joined the queue.`
      );

      // Validate that this player actually exists in the DB
      const [player, created] = await Player.findOrCreate({
        where: { playerId },
        defaults: { playerId: playerId }
      });

      // If no one is waiting for this language, store this player
      if (!waitingPlayers[language]) {
        waitingPlayers[language] = {
          socketId: socket.id,
          playerId: playerId,
        };

        // Emit to this socket that it is waiting
        socket.emit('waitingForOpponent');

        console.log(`No one waiting yet for ${language}; stored ${socket.id} as waiting.`);

        // IMPORTANT: Broadcast updated waiting count to everyone
        io.emit('lobbyCountUpdate', getWaitingPlayersCount());

      } else {
        // We have a waiting player for the same language -> form a new game
        const waitingSocketId = waitingPlayers[language].socketId;
        const waitingPlayerId = waitingPlayers[language].playerId;

        const gameId = uuidv4();
        console.log(
          `Forming game ${gameId} for language=${language} between ${waitingSocketId} and ${socket.id}`
        );

        // Create a new game in DB
        const newGame = await Game.create({
          gameId,
          player1Id: waitingPlayerId,
          player2Id: playerId,
          botId: null,  // null means not playing vs. bot
          language,
          roundCount: 0,
          status: "in_progress",
          wordsArray: JSON.stringify([]),
        });

        // Save in-memory references
        activeGames[gameId] = {
          player1Socket: waitingSocketId,
          player1Id: waitingPlayerId,
          player2Socket: socket.id,
          player2Id: playerId,
          roundWords: { player1: null, player2: null },
        };

        // Notify both players that the game has started
        io.to(waitingSocketId).emit('gameStarted', {
          gameId,
          role: 'player1',
          opponentSocket: socket.id,
        });
        io.to(socket.id).emit('gameStarted', {
          gameId,
          role: 'player2',
          opponentSocket: waitingSocketId,
        });

        // Clear the waiting slot for this language (the 2 players now have a game)
        delete waitingPlayers[language];

        // Also broadcast the updated waiting count
        io.emit('lobbyCountUpdate', getWaitingPlayersCount());
      }
    });

    /**
     * 2) Handle a player submitting a new word (human)
     */
    socket.on('submitWordHuman', async (data) => {
      // data = { gameId, word, role }
      const { gameId, word, role } = data;
      const gameObj = activeGames[gameId];
      if (!gameObj) {
        console.log(`No active game found in memory with id: ${gameId}`);
        return;
      }

      // Validate role
      if (
        (role === 'player1' && socket.id !== gameObj.player1Socket) ||
        (role === 'player2' && socket.id !== gameObj.player2Socket)
      ) {
        console.log(`Socket ${socket.id} tried to submit word with invalid role.`);
        return;
      }

      // Check if this player has already played
      if (gameObj.roundWords[role] !== null) {
        console.log(`${role} already played this round. Ignoring duplicate submission.`);
        return;
      }

      // Store the word
      gameObj.roundWords[role] = word;
      console.log(`Game ${gameId}: ${role} submitted "${word}"`);

      // Check if both players have played
      const bothPlayed =
        gameObj.roundWords.player1 !== null && gameObj.roundWords.player2 !== null;

      if (bothPlayed) {
        // Grab both words
        const p1Word = gameObj.roundWords.player1;
        const p2Word = gameObj.roundWords.player2;

        let status = "in_progress";

        if (p1Word.toLowerCase() === p2Word.toLowerCase()) {
          status = "won";
          console.log(`Game ${gameId}: Both players submitted the same word!`);
        } else if (gameObj.roundWords.length > MAX_NUMBER_OF_ROUNDS) {
          status = "lost";
        }

        // Send roundResult to player1
        io.to(gameObj.player1Socket).emit('roundResult', {
          yourWord: p1Word,
          opponentWord: p2Word,
          status: status,
        });
        // Send roundResult to player2
        io.to(gameObj.player2Socket).emit('roundResult', {
          yourWord: p2Word,
          opponentWord: p1Word,
          status: status,
        });

        // Update DB
        try {
          const game = await Game.findByPk(gameId);
          let wordsPlayed1 = game.wordsPlayed1 ? JSON.parse(game.wordsPlayed1) : [];
          let wordsPlayed2 = game.wordsPlayed2 ? JSON.parse(game.wordsPlayed2) : [];
          wordsPlayed1.push(p1Word);
          wordsPlayed2.push(p2Word);

          game.wordsPlayed1 = JSON.stringify(wordsPlayed1);
          game.wordsPlayed2 = JSON.stringify(wordsPlayed2);
          game.roundCount = game.roundCount + 1;
          game.status = status;
          await game.save();
        } catch (err) {
          console.error('Error updating game in DB:', err);
        }

        // Reset roundWords so a new round can begin
        gameObj.roundWords.player1 = null;
        gameObj.roundWords.player2 = null;
      }
    });

    /**
     * 3) Handle player disconnect
     */
    socket.on('disconnect', () => {
      console.log('User disconnected:', socket.id);

      // If this user was in the waiting list, remove them
      for (const lang in waitingPlayers) {
        if (waitingPlayers[lang].socketId === socket.id) {
          console.log(`Removing waiting player for language ${lang} due to disconnect`);
          delete waitingPlayers[lang];

          // Update waiting count
          io.emit('lobbyCountUpdate', getWaitingPlayersCount());
        }
      }

      // Also remove them from any active games
      Object.keys(activeGames).forEach((gId) => {
        const g = activeGames[gId];
        if (g.player1Socket === socket.id || g.player2Socket === socket.id) {
          console.log(`Removing game ${gId} due to disconnect`);
          delete activeGames[gId];
          // Optionally notify the other player that the game ended
          const otherSocket = g.player1Socket === socket.id ? g.player2Socket : g.player1Socket;
          io.to(otherSocket).emit('opponentDisconnected');
        }
      });
    });
  });
}