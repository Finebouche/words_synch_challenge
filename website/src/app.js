import express from 'express';
import path, {dirname} from 'path';
import axios from 'axios';
import {Game, Player} from './database.js'; // Make sure to update the extension
import OpenAI from 'openai';
import fs from 'fs';
import {fileURLToPath} from 'url';

// load openai key from open_ai_key.txt
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || fs.readFileSync(path.join(__dirname, 'open_ai_key.txt'), 'utf8').trim();
const HUGGINGFACE_API_TOKEN = process.env.HUGGINGFACE_API_TOKEN || fs.readFileSync(path.join(__dirname, 'huggingface_api_token.txt'), 'utf8').trim();
const openaiClient = new OpenAI({apiKey: OPENAI_API_KEY});

const app = express();
// Serve all static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "index.html"));
});
app.use(express.json());


// USER AUTHENTICATION
app.post('/login', async (req, res) => {
    const { playerId } = req.body;
    try {
        const player = await Player.findByPk(playerId);
        if (player) {
            res.json({ pseudonym: player.pseudonym, playerId: player.playerId});
        } else {
            res.status(404).send('User not found');
        }
    } catch (error) {
        res.status(500).send('Server error');
    }
});

app.post('/create', async (req, res) => {
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

app.post('/update-pseudonym', async (req, res) => {
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

// END USER AUTHENTICATION


// MODEL INTERACTION   

// List of available models (has to be inferior to 5B parameters for the API to work)
const availableModels = [
    { name: 'gpt-3.5-turbo', type: 'chat-completion', languages: ['en', 'fr'], provider: "openai"},
    { name: 'gpt-4', type: 'chat-completion', languages: ['en', 'fr'], provider: "openai" },
    { name: 'openai-community/gpt2', type: 'text-generation', languages : ['en', 'fr'], provider: "huggingface" },
    { name: 'google/flan-t5-large', type: 'text2text-generation' , languages : ['en'], provider: "huggingface"  },
    { name: 'google-bert/bert-base-uncased', type: 'fill-mask' , languages : ['en'] , mask_token: '[MASK]', provider: "huggingface" },
    { name: 'FacebookAI/xlm-roberta-base', type: 'fill-mask' , languages : ['es', 'en','fr'], mask_token: "<mask>", provider: "huggingface" },
    // Add other models here
];
// Endpoint to get available models
app.get('/available-models', (req, res) => {
    res.json(availableModels);
});


app.post('/initialize-model', async (req, res) => {
    const model = req.body.model; // Retrieve the model from the request body
    const playerId = req.body.player_id;

    // Check if the model name is valid
    const modelNames = availableModels.map(model => model.name);
    if (!modelNames.includes(model.name)) {
        return res.status(400).send("Invalid model name");
    }

    let token = "initialisation"
    let parameters;
    if (model.type === 'text2text-generation') {
    }

    if (model.type === 'text-generation') {
        parameters = {return_full_text:false, max_new_tokens: 6 }
    }

    if (model.type === 'fill-mask') {
        token = token + " " + model.mask_token
    }

    // If new player doesn't exist in the database, create a new player
    const [player, created] = await Player.findOrCreate({
        where: { playerId: playerId },
        defaults: { playerId: playerId }
    });

    const newGame = await Game.create({
        botId: model.name,
        playerId: playerId,
    });


    if (model.provider === "huggingface") {
        try {
              const response = await axios.post(
                    `https://api-inference.huggingface.co/models/${model.name}`,
                    { inputs: token },
                    {
                        headers: { Authorization: `Bearer ${HUGGINGFACE_API_TOKEN}` }
                    }
                );

            res.json({ gameData: response.data, gameId: newGame.gameId });
        } catch (error) {
            console.error("Error calling the Hugging Face API", error);
            if (error.response && error.response.status === 503) {
                res.status(503).send("Model is loading");
            } else {
                res.status(500).send("Error calling the Hugging Face API");
            }
        }
    } else if (model.provider === "openai") {
        // check the availability of openai api :
        try {
            const response = await openaiClient.chat.completions.create({
                model: model.name,
                messages: [{ role: "system", content: "Hello" }],
                max_tokens: 20,
                temperature: 1.2,
            });
            res.json({ gameData: response.data, gameId: newGame.gameId });
        } catch (error) {
            console.error("Error calling the OpenAI API", error.response ? error.response.data : error);
            res.status(500).send("Error calling the OpenAI API");
        }
    }
});

const RULE_TOKEN = "You are playing a game where at each round both player say a word. The goal is to produce the " +
    "same word based on previous words at which point the game ends."

const ROUND_ONE = "Round 1. New game, please give your first (really random) word and only that word."

const huggingFaceRoundTemplate = (roundNumber, pastWords) => {
    return `\nRound ${roundNumber}! Past words, forbidden to use are ${pastWords.join(', ')}. Please give your word for the current round.\n`;
};

async function huggingfacecall(model, round, past_words_array, res) {
    // Construct the rounds history based on past words
    let interactionHistory = RULE_TOKEN + ROUND_ONE
    if (Array.isArray(past_words_array) && past_words_array.length > 0) {
        for (let i = 0; i < past_words_array.length; i++) {
            if (i % 2 === 0) {  // Start a new round every two words
                interactionHistory += huggingFaceRoundTemplate(Math.floor(i / 2) + 2, past_words_array.slice(0, i));
            interactionHistory += `Player ${i % 2 + 1}: '${past_words_array[i]}'\n`;
            }
        }
    }

    // Example of gameplay
    let EXAMPLES = "\n\nExample of gameplay 1:\n" +
        RULE_TOKEN +
        ROUND_ONE +
        "Player 1: 'Apple'\n" +
        "Player 2: 'Banana'\n\n" +
        huggingFaceRoundTemplate(2, ['Apple', 'Banana']) +
        "Player 1 (Thinking): 'Apple' and 'Banana' are both fruits. I'll abstract these to their category to see if we can align. 'Fruit' should work." +
        "Player 1: 'Fruit'\n" +
        "Player 2 (Thinking): 'Apple' and 'Banana' can both be yellow, therefore I should say 'Yellow'." +
        "Player 2: 'Yellow'\n\n" +
        huggingFaceRoundTemplate(3, ['Apple', 'Banana', 'Fruit', 'Yellow']) +
        "Player 1 (Thinking): 'Yellow'... what’s a fruit that fits this color? 'Lemon' is perfect." +
        "Player 1: 'Lemon'\n" +
        "Player 2 (Thinking): 'Yellow' fruit... 'Lemon' is the first that comes to mind." +
        "Player 2: 'Lemon'\n\n" +
        "The game is WON as both players said the same word: 'Vegetable'.\n\n" +
        "\n\nExample of gameplay 2:\n" +
        RULE_TOKEN +
        ROUND_ONE +
        "Player 1: 'House'\n" +
        "Player 2: 'Mountain'\n\n" +
        huggingFaceRoundTemplate(2, ['House', 'Mountain']) +
        "Player 1 (Thinking): 'House' suggests structure, 'Mountain' suggests a natural setting. A 'Monastery' often combines these ideas, nestled in mountains." +
        "Player 1: 'Monastery'\n" +
        "Player 2 (Thinking): 'Mountain' makes me think of natural, secluded places. A 'Cave' fits this theme well." +
        "Player 2: 'Cave'\n\n" +
        huggingFaceRoundTemplate(3, ['House', 'Mountain', 'Monastery', 'Cave']) +
        "Player 1 (Thinking): 'Monastery' and 'Cave' both evoke a sense of secrecy or hidden qualities. I'll say 'Secret' to capture that essence." +
        "Player 1: 'Secret'\n" +
        "Player 2 (Thinking): 'Secret' makes me think of something mystical or hidden. Wait, 'Monastery' fits this theme perfectly... " +
        "Player 2: 'Monastery'\n\n" +
        "The game is LOST as Player 2 gave a previously given word.\n\n";


    let token;
    let parameters;  // https://huggingface.co/docs/api-inference/detailed_parameters
    let currentRoundPrompt;

    if (round === 1) {
        currentRoundPrompt = ROUND_ONE;
    } else {
        currentRoundPrompt =`\nRound ${round}! Past words, forbidden to use are ${past_words_array.join(', ')}. Please give your word for the current round.\n`;
    }

    console.log(model.type)

    if (model.type === 'text2text-generation') {
        token = EXAMPLES + interactionHistory + currentRoundPrompt + "Player 1 : '"
    } else if (model.type === 'text-generation') {
        token = EXAMPLES + interactionHistory + currentRoundPrompt + "Player 1 : '"
        parameters = {return_full_text: false, max_new_tokens: 20}
    } else if (model.type === 'fill-mask') {
        token = EXAMPLES + interactionHistory + currentRoundPrompt + "Player 1 : '" + model.mask_token + "'\n."
    } else {
        return res.status(400).send("Invalid model type");
    }

    try {
        const response = await axios.post(
            `https://api-inference.huggingface.co/models/${model.name}`,
            {inputs: token, parameters: parameters, options: {wait_for_model: true}},
            {headers: {Authorization: `Bearer ${HUGGINGFACE_API_TOKEN}`},}
        );

        let llmWord
        console.log(response.data)

        if (model.type === "fill-mask") {
            llmWord = response.data[0].token_str;
        } else {
            llmWord = response.data[0].generated_text.match(/\b\w+\b/)?.[0];
        }
        return llmWord;

    } catch (error) {
        console.error("Error calling the Hugging Face API", error);
        res.status(500).send("Error calling the Hugging Face API");
    }
}

async function openaicall(model, round, past_words_array, res) {
    // Initialize messages
    let messages = [];
    messages.push({role: "system", content: RULE_TOKEN});
    messages.push({role: "system", content: "Round 1. New game, please give your first (really random) word and only that word."});

    function openAIRoundTemplate(round_number, past_words_array, word) {
        if (round === 1) {
            return "Round 1. New game, please give your first (really random) word and only that word."
        } else {
            return (
                `${word}! We said different words, let's do another round then and try to get closer. Past words,  ` +
                `forbidden to use are [${past_words_array.join(', ')}]. Please give only your word for this round and I will ` +
                "give you mine."
            )
        }
    }

    // If there are past words, reconstruct the conversation
    if (past_words_array && past_words_array.length > 0 && round > 1) {
        for (let i = 0; i < past_words_array.length; i++) {
            if (i % 2 === 0) {
                // Bot's word (Player 1)
                messages.push({role: "assistant", content: `'${past_words_array[i]}'`});
            } else {
                // Player's word (Player 2)
                messages.push({role: "user", content: openAIRoundTemplate(round, past_words_array, past_words_array[i])});
            }
        }
    }
    try {
        console.log(messages)
        const response = await openaiClient.chat.completions.create({
            model: model.name,
            messages: messages,
            max_tokens: 20,
            temperature: 1.2,
        });

        const llmWord = response.choices[0].message.content.trim();
        console.log(response.choices[0].message)

        // Simple regex to extract the word (remove any non-alphabetic characters)
        return llmWord.replace(/[^a-zA-Z]/g, "");

    } catch (error) {
        console.error("Error calling the OpenAI API", error.response ? error.response.data : error);
        res.status(500).send("Error calling the OpenAI API");
    }
}

app.post('/query-model', async (req, res) => {
    // Doc at https://huggingface.co/docs/api-inference/detailed_parameters?code=js
    const past_words_array = req.body.previous_words;
    const model = req.body.model;
    const gameId = req.body.game_id;
    const newWord = req.body.new_word;
    
    // Check if the model name is valid
    const modelNames = availableModels.map(model => model.name);
    if (!model || typeof model.name !== 'string' || !modelNames.includes(model.name)) {
        return res.status(400).send("Invalid model name");
    }
    // Compute the round number
    const round = Math.floor(past_words_array.length / 2) + 1;


    let llmWord
    if (model.provider === "huggingface") {
        llmWord = await huggingfacecall(model, round, past_words_array, res)
    } else if (model.provider === "openai") {
        llmWord = await openaicall(model, round, past_words_array, res)
    } else {
        return res.status(400).send("Invalid model provider");
    }

    if (past_words_array.includes(newWord) || past_words_array.includes(llmWord) || round > 5) {
        return res.json({llmWord: llmWord, status: "loses"});
    } else if (llmWord === newWord) {
        return res.json({llmWord: llmWord, status: "wins"});
    } else {
        res.json({llmWord: llmWord, status: "continue"});
    }

    let game = await Game.findByPk(gameId);
    let wordsArray = game.wordsArray ? JSON.parse(game.wordsArray) : [];
    wordsArray.push(llmWord);
    wordsArray.push(newWord);
    await game.update({wordsArray: JSON.stringify(wordsArray), roundCount: round, gameWon: llmWord === newWord});
});
// END MODEL INTERACTION

// Export app as the default export
export default app;