
const { Sequelize, DataTypes } = require('sequelize');
const { v4: uuidv4 } = require('uuid'); // Correctly import uuidv4

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
  }
});

const Game = sequelize.define('Game', {
  gameId: {
    type: DataTypes.UUID,
    defaultValue: () => uuidv4(), 
    primaryKey: true
  },
  playerId: {
    type: DataTypes.INTEGER,
    allowNull: false,
    references: {
      model: 'Players',
      key: 'playerId'
    }
  },
  botId: {
    type: DataTypes.INTEGER,
    allowNull: false
  },
  roundCount: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 0
  },
  gameWon: {
    type: DataTypes.BOOLEAN,
    allowNull: false,
    defaultValue: false
  },
  wordsArray: {
    type: DataTypes.TEXT,
    allowNull: false,
    defaultValue: '[]'
  },
});

Player.hasMany(Game, {foreignKey: 'playerId'});
Game.belongsTo(Player, {foreignKey: 'playerId'});

const initDb = async () => {
  await sequelize.sync(); // Sync models to the database
  console.log('Database synced!');
};

module.exports = { initDb, Player, Game };