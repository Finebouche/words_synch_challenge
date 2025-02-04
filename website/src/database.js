import { Sequelize, DataTypes } from 'sequelize';
import { v4 as uuidv4 } from 'uuid';

// Setup Sequelize connection
const sequelize = new Sequelize({
  dialect: 'sqlite',
  storage: 'word_sync.db', // Path to the SQLite database file
  logging: false // You can enable logging by setting it to console.log
});

// Define models
const Player = sequelize.define('Player', {
  playerId: {
    type: DataTypes.STRING,
    allowNull: false,
    primaryKey: true,
    unique: true
  },
  pseudonym: {
    type: DataTypes.STRING,
    allowNull: true
  },
  ageGroup: {
    type: DataTypes.STRING,
    allowNull: true
  },
  gender: {
    type: DataTypes.STRING,
    allowNull: true
  },
  region: {
    type: DataTypes.STRING,
    allowNull: true
  },
  llmKnowledge: {
    type: DataTypes.STRING,
    allowNull: true
  }
});

const Game = sequelize.define('Game', {
  gameId: {
    type: DataTypes.UUID,
    defaultValue: () => uuidv4(),
    primaryKey: true
  },
  player1Id: {
    type: DataTypes.STRING,
    allowNull: false,
    references: {
      model: 'Players',
      key: 'playerId'
    }
  },
  player2Id: {
    type: DataTypes.STRING,
    allowNull: true, // Nullable since a game might involve a bot
    references: {
      model: 'Players',
      key: 'playerId'
    }
  },
  botId: {
    type: DataTypes.INTEGER,
    allowNull: true, // Nullable since a game might involve another human
  },
  language: {
    type: DataTypes.STRING,
    allowNull: false,
    defaultValue: 'en' // Defaulting to English; adjust based on your most common language if necessary
  },
  roundCount: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 0
  },
  status: {
    type: DataTypes.STRING,
    allowNull: false,
    defaultValue: 'in_progress',
    validate: {
      isIn: [['won', 'lost', 'in_progress']]
    },
  },
  wordsPlayed1: {
    type: DataTypes.TEXT,
    allowNull: false,
    defaultValue: '[]'
  },
  wordsPlayed2: {
    type: DataTypes.TEXT,
    allowNull: false,
    defaultValue: '[]'
  },
  surveyAnswers: {
    type: DataTypes.TEXT,
    allowNull: true,        // Let it be null if no answers yet
    defaultValue: '[]'
  },
});

Player.hasMany(Game, {
  as: 'gamesAsPlayer1',
  foreignKey: 'player1Id'
});
Player.hasMany(Game, {
  as: 'gamesAsPlayer2',
  foreignKey: 'player2Id'
});
Game.belongsTo(Player, {
  as: 'player1',
  foreignKey: 'player1Id'
});
Game.belongsTo(Player, {
  as: 'player2',
  foreignKey: 'player2Id'
});
Game.addHook('beforeValidate', (game, options) => {
  if (game.player2Id && game.botId) {
    throw new Error('A game cannot have both a second player and a bot.');
  }
});

const initDb = async () => {
  await sequelize.sync(); // Sync models to the database
  console.log('Database synced!');
};

export { initDb, Game, Player };

export default sequelize;