import express from 'express';
import path, { dirname } from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import axios from 'axios';
import { Game, Player } from '../database.js';
import OpenAI from 'openai';

const router = express.Router();

// Setup __dirname for this file
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load keys
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || fs.readFileSync(path.join(__dirname, '..', 'open_ai_key.txt'), 'utf8').trim();
const HUGGINGFACE_API_TOKEN = process.env.HUGGINGFACE_API_TOKEN || fs.readFileSync(path.join(__dirname, '..', 'huggingface_api_token.txt'), 'utf8').trim();
const openaiClient = new OpenAI({ apiKey: OPENAI_API_KEY });

const MAX_NUMBER_OF_ROUNDS = 15;

// MODEL INTERACTION
const availableModels = [
    // OPENAI
    { name: 'gpt-4o', type: 'chat-completion', languages: ['en', 'fr'], provider: "openai", disabled: false },
    { name: 'gpt-4o-mini', type: 'chat-completion', languages: ['en', 'fr'], provider: "openai", disabled: true },
    { name: 'gpt-4', type: 'chat-completion', languages: ['en', 'fr'], provider: "openai", disabled: true },
    // HUGGINGFACE
    { name: 'meta-llama/Llama-3.2-3B', type: 'text-generation' , languages : ['en','fr', 'es'], provider: "huggingface", disabled: true  },
    { name: 'meta-llama/Llama-3.2-1B', type: 'text-generation' , languages : ['en','fr', 'es'], provider: "huggingface", disabled: true  },
    { name: 'openai-community/gpt2', type: 'text-generation', languages : ['en', 'fr'], provider: "huggingface", disabled: true  },
    { name: 'google/flan-t5-large', type: 'text2text-generation' , languages : ['en'], provider: "huggingface", disabled: true   },
    { name: 'google-bert/bert-base-uncased', type: 'fill-mask' , languages : ['en'] , mask_token: '[MASK]', provider: "huggingface", disabled: true  },
    { name: 'distilbert/distilroberta-base', type: 'fill-mask' , languages : ['en'] , mask_token: "<mask>", provider: "huggingface", disabled: true  },
];

// Endpoint to get available models
router.get('/available-models', (req, res) => {
    res.json(availableModels);
});

async function checkIfWordExists(llmWord) {
        const endpoint = `https://en.wiktionary.org/w/api.php`;

        // Helper function to create parameters and make API request
        async function fetchWordInfo(variant) {
            const params = new URLSearchParams({
                action: 'query',
                format: 'json',
                titles: variant,
                origin: '*'
            });
            const url = `${endpoint}?${params.toString()}`;
            const response = await fetch(url);
            return response.json();
        }

        // Fetch with the word entirely in lowercase
        const lowerCaseData = await fetchWordInfo(llmWord.toLowerCase());
        const firstCapData = await fetchWordInfo(llmWord.charAt(0).toUpperCase() + llmWord.slice(1).toLowerCase());

        // Check for valid pages in response data
        if (!lowerCaseData.query.pages['-1'] || !firstCapData.query.pages['-1']) {
            return true;
        } else {
            return false;
        }
}

router.post('/initialize-model', async (req, res) => {
    const model = req.body.model; // Retrieve the model from the request body
    const playerId = req.body.player_id;
    const language = req.body.language;

    const modelNames = availableModels.map(m => m.name);
    if (!modelNames.includes(model.name)) {
        return res.status(400).send("Invalid model name");
    }

    let token = "initialisation"
    let parameters;

    if (model.type === 'text2text-generation') {
        // Could specify parameters if needed
    }

    if (model.type === 'text-generation') {
        parameters = {return_full_text:false, max_new_tokens: 6 }
    }

    if (model.type === 'fill-mask') {
        token = token + " " + model.mask_token
    }

    const [player, created] = await Player.findOrCreate({
        where: { playerId: playerId },
        defaults: { playerId: playerId }
    });

    const newGame = await Game.create({
        botId: model.name,
        player1Id: playerId,
        language: language,
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

const RULE_TOKEN = "You are a helpful assistant playing a game where at each round both player say a word. The goal is to produce the same word than the other player based on previous words of the game."
const ROUND_ONE = "Round 1. New game, please give your first (really random) word and only that word."

const huggingFaceRoundTemplate = (roundNumber, pastWords) => {
    return `\nRound ${roundNumber}! Past words, forbidden to use are ${pastWords.join(', ')}. Please give your word for the current round.\n`;
};

async function huggingfacecall(model, round, past_words_array, res) {
    let interactionHistory = RULE_TOKEN + ROUND_ONE
    if (Array.isArray(past_words_array) && past_words_array.length > 0) {
        for (let i = 0; i < past_words_array.length; i++) {
            if (i % 2 === 0) {
                interactionHistory += huggingFaceRoundTemplate(Math.floor(i / 2) + 2, past_words_array.slice(0, i));
                interactionHistory += `Player ${i % 2 + 1}: '${past_words_array[i]}'\n`;
            }
        }
    }

    let EXAMPLES = "\n\nExample of gameplay 1:\n" +
        RULE_TOKEN +
        ROUND_ONE +
        "Player 1: 'Apple'\n" +
        "Player 2: 'Banana'\n\n" +
        huggingFaceRoundTemplate(2, ['Apple', 'Banana']) +
        "Player 1 (Thinking): 'Apple' and 'Banana' are both fruits... Player 1: 'Fruit'\n" +
        "Player 2 (Thinking): ... Player 2: 'Yellow'\n\n" +
        huggingFaceRoundTemplate(3, ['Apple', 'Banana', 'Fruit', 'Yellow']) +
        "Player 1 (Thinking): 'Lemon'... Player 1: 'Lemon'\n" +
        "Player 2 (Thinking): 'Lemon'... Player 2: 'Lemon'\n\n" +
        "The game is WON.\n\n" +
        "\n\nExample of gameplay 2:\n" +
        RULE_TOKEN +
        ROUND_ONE +
        "Player 1: 'House'\n" +
        "Player 2: 'Mountain'\n\n" +
        huggingFaceRoundTemplate(2, ['House', 'Mountain']) +
        "Player 1: 'Monastery'\n" +
        "Player 2: 'Cave'\n\n" +
        huggingFaceRoundTemplate(3, ['House', 'Mountain', 'Monastery', 'Cave']) +
        "Player 1: 'Secret'\n" +
        "Player 2: 'Monastery'\n\n" +
        "The game is LOST as Player 2 repeated a word.\n\n";

    let token;
    let parameters;
    let currentRoundPrompt;

    if (round === 1) {
        currentRoundPrompt = ROUND_ONE;
    } else {
        currentRoundPrompt = `\nRound ${round}! Past words, forbidden to use are ${past_words_array.join(', ')}. Please give your word for the current round.\n`;
    }

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
            { inputs: token, parameters: parameters, options: { wait_for_model: true } },
            { headers: { Authorization: `Bearer ${HUGGINGFACE_API_TOKEN}` } }
        );

        let llmWord;
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
    let messages = [];
    messages.push({role: "developer", content: RULE_TOKEN});
    messages.push({role: "user", content: ROUND_ONE});

    function openAIRoundTemplate(round_number, past_words_array, word) {
        if (round_number === 1) {
            return "Round 1. New game, please give your first (really random) word and only that word. Be really creative please."
        } else {
            return (
                `${word}! We said different words, let's do another round. So far we have used the words: [${past_words_array.join(', ')}], they are now forbidden` +
                "What do you think my next word is going to be ?"
            )
        }
    }

    if (past_words_array && past_words_array.length > 0 && round > 1) {
        for (let i = 0; i < past_words_array.length; i++) {
            if (i % 2 === 0) {
                messages.push({role: "assistant", content: `'${past_words_array[i]}'`});
                console.log(past_words_array[i])
            } else {
                messages.push({role: "user", content: openAIRoundTemplate(round, past_words_array, past_words_array[i])});
            }
        }
    }

    try {
        let temp = round === 1 ? 2.0 : 1.2;
        const response = await openaiClient.chat.completions.create({
            model: model.name,
            messages: messages,
            max_tokens: 15,
            temperature: temp,
        });

        const llmWord = response.choices[0].message.content.trim();
        return llmWord.replace(/[^a-zA-Z]/g, "");

    } catch (error) {
        console.error("Error calling the OpenAI API", error.response ? error.response.data : error);
        res.status(500).send("Error calling the OpenAI API");
    }
}

router.post('/query-model', async (req, res) => {
    const past_words_array = req.body.previous_words;
    const model = req.body.model;
    const gameId = req.body.game_id;
    const newWord = req.body.new_word;

    const modelNames = availableModels.map(m => m.name);
    if (!model || typeof model.name !== 'string' || !modelNames.includes(model.name)) {
        return res.status(400).send("Invalid model name");
    }

    const round = Math.floor(past_words_array.length / 2) + 1;

    let llmWord;
    while (true) {
        if (model.provider === "huggingface") {
            llmWord = await huggingfacecall(model, round, past_words_array, res);
        } else if (model.provider === "openai") {
            llmWord = await openaicall(model, round, past_words_array, res);
        } else {
            return res.status(400).send("Invalid model provider");
        }
        console.log(`Model ${model.name} returned word: ${llmWord}`);

        // Ensure the function waits for the check before continuing
        let exists = await checkIfWordExists(llmWord);

        if (exists) {
            llmWord = llmWord.replace(/[^a-zA-Z]/g, ""); // Fix replace issue
            break; // Exit loop when we get a unique word
        }
    }

    let status = "in_progress";
    if (llmWord.toLowerCase() === newWord.toLowerCase()) {
        status = "won";
    }
    else if (past_words_array.includes(newWord) || past_words_array.includes(llmWord) || round > MAX_NUMBER_OF_ROUNDS) {
        status = "lost";
    }

    let game = await Game.findByPk(gameId);
    let wordsArray = game.wordsArray ? JSON.parse(game.wordsArray) : [];
    wordsArray.push(llmWord);
    wordsArray.push(newWord);
    await game.update({
        wordsArray: JSON.stringify(wordsArray),
        roundCount: round,
        gameWon: llmWord === newWord,
        status: status
    });

    res.json({llmWord: llmWord, status: status});
});

export default router;