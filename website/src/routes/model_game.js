import express from 'express';
import path, { dirname } from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import axios from 'axios';
import { Game, Player } from '../database.js';
import OpenAI from 'openai';
import { HfInference } from "@huggingface/inference";

const router = express.Router();

// Setup __dirname for this file
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load keys
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || fs.readFileSync(path.join(__dirname, '..', 'open_ai_key.txt'), 'utf8').trim();
const HUGGINGFACE_API_TOKEN = process.env.HUGGINGFACE_API_TOKEN || fs.readFileSync(path.join(__dirname, '..', 'huggingface_api_token.txt'), 'utf8').trim();
const openaiClient = new OpenAI({ apiKey: OPENAI_API_KEY });
const client = new HfInference(HUGGINGFACE_API_TOKEN);

const MAX_NUMBER_OF_ROUNDS = 15;

///////////////////////
//  INITIALISATION
///////////////////////


const availableModels = [
    // OPENAI
    { name: 'gpt-4o', type: 'chat-completion', languages: ['en', 'fr'], provider: "openai", disabled: false },
    { name: 'gpt-4o-mini', type: 'chat-completion', languages: ['en', 'fr'], provider: "openai", disabled: false },
    { name: 'gpt-4', type: 'chat-completion', languages: ['en', 'fr'], provider: "openai", disabled: false },
    // HUGGINGFACE
    { name: 'google/gemma-2-2b-it', type: 'chat-completion' , languages : ['en','fr', 'es'], provider: "hf-inference", disabled: false  },
    { name: 'meta-llama/Llama-3.2-1B', type: 'chat-completion' , languages : ['en','fr', 'es'], provider: "hf-inference", disabled: false  },
    { name: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', type: 'chat-completion', languages : ['en', 'fr'], provider: "hf-inference", disabled: false  },
    { name: 'microsoft/phi-4', type: 'chat-completion' , languages : ['en'], provider: "nebius", disabled: false   },
    { name: 'deepseek-ai/DeepSeek-V3-0324', type: 'chat-completion' , languages : ['en'] , provider: "fireworks-ai", disabled: false  },
    { name: 'google/flan-t5-large', type: 'text2text-generation' , languages : ['en'], provider: "hf-inference", disabled: true   },
    { name: 'google-bert/bert-base-uncased', type: 'fill-mask' , languages : ['en'] , mask_token: '[MASK]', provider: "hf-inference", disabled: true  },
    { name: 'FacebookAI/xlm-roberta-base', type: 'fill-mask' , languages : ['en'] , mask_token: "<mask>", provider: "hf-inference", disabled: false  },
];

// Endpoint to get available models
router.get('/available-models', (req, res) => {
    res.json(availableModels);
});

router.post('/initialize-model', async (req, res) => {
    const model = req.body.model; // Retrieve the model from the request body
    const playerId = req.body.player_id;
    const language = req.body.language;
    const gameConfig = req.body.game_config;
    const gameConfigOrder = req.body.game_config_order;

    // Initialize player and game
    const [player, created] = await Player.findOrCreate({
        where: { playerId: playerId },
        defaults: { playerId: playerId, gameConfigOrder: gameConfigOrder }
    });

    const newGame = await Game.create({
        botId: model.name,
        player1Id: playerId,
        language: language,
        gameConfigPlayer1: gameConfig,
        trueGameConfig: "human_vs_bot",
    });

    // Initialize LLM

    const modelNames = availableModels.map(m => m.name);
    if (!modelNames.includes(model.name)) {
        return res.status(400).send("Invalid model name");
    }

    let token = "initialisation"
    let parameters;

    if (model.provider === "openai") {
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
    } else {

        if (model.type === 'text2text-generation') {
            // Could specify parameters if needed
        } else if (model.type === 'text-generation') {
            parameters = {return_full_text:false, max_new_tokens: 6 }
        } else if (model.type === 'fill-mask') {
            token = token + " " + model.mask_token
        } else if (model.type === 'chat-completion') {
        }

        try {
            const response = await client.textGeneration({
                inputs : token,
                model: model.name,
                provider: model.provider,
            });

            res.json({ gameData: response.data, gameId: newGame.gameId });
        } catch (error) {
            console.error("Error calling the Hugging Face API", error);
            if (error.response && error.response.status === 503) {
                res.status(503).send("Model is loading");
            } else {
                res.status(500).send("Error calling the Hugging Face API");
            }
        }
    }

});

///////////////////////
// GAME INTERACTIONS
///////////////////////

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

function checkIfWordPreviouslyUsed(newWord, pastWords) {
    // iterate through past words and check if the new word has been used before
    for (let i = 0; i < pastWords.length; i++) {
        if (newWord.toLowerCase() === pastWords[i].toLowerCase()) {
            return true;
        }
    }
}

const RULE_TOKEN = "You are a helpful assistant playing a game where at each round both player write a word. " +
    "The goal is to produce the same word than the other player based on previous words of the game."
const ROUND_ONE = "Round 1. New game, please give your first (really random) word and only that word. You can be a bit creative but not too much. Be sure to finish your answer with it"

function roundTemplate(round_number, past_words_array, player_word, bot_word) {
    if (round_number === 1) {
        return ROUND_ONE;
    } else {
        return (
            `${player_word}! We said different words, let's do another round. So far we have used the words: [${past_words_array.join(', ')}], they are now forbidden` +
            `Based on previous words, what word would be most likely for next round given that my word was ${player_word} and your word was ${bot_word}` +
            "Please give only your word for this round."
        )
    }
}

async function huggingfacecall(model, round, past_words_array, res) {

    const huggingFaceRoundTemplate = (roundNumber, pastWords) => {
        return `\nRound ${roundNumber}! Past words, forbidden to use are ${pastWords.join(', ')}. Please give your word for the current round.\n`;
    };

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
        const response = await client.textGeneration({
            model: model.name,
            provider: model.provider,
            inputs: token,
            parameters: parameters,
            options: { wait_for_model: true }
        });

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

async function chatCompletioncall(model, round, past_words_array, res) {
    let messages = [];
    messages.push({role: "assistant", content: RULE_TOKEN});
    messages.push({role: "user", content: ROUND_ONE});

    if (past_words_array && past_words_array.length > 0 && round > 1) {
        for (let i = 0; i < past_words_array.length; i++) {
            if (i % 2 === 0) {
                messages.push({role: "assistant", content: `'${past_words_array[i]}'`});
                // console.log(past_words_array[i]) // that was to check that it was the right index (and it is)
            } else {
                messages.push({role: "user", content: roundTemplate(round, past_words_array.slice(0, i), past_words_array[i], past_words_array[i - 1])});
            }
        }
    }

    let response;
    try {
        let temp = round === 1 ? 1.6 : 1.1;
        let max_tokens = round === 1 ? 50 : 20;
        console.log("Max tokens: ", max_tokens);
        console.log("Temperature: ", temp);
        console.log("Messages: ", messages);
        if (model.provider === "openai") {
            response = await openaiClient.chat.completions.create({
                model: model.name,
                messages: messages,
                max_tokens: max_tokens,
                temperature: temp,
            });
        } else {
            response = await client.chatCompletion({
                model: model.name,
                provider: model.provider,
                messages: messages,
                max_tokens: max_tokens,
                temperature: temp,
            });
        }

        const fullText = response.choices[0].message.content.trim();
        console.log("Full reply: ", fullText);
        // Split by whitespace
        const tokens = fullText.split(/\s+/);
        // The last token is what we want:
        let lastWord = tokens[tokens.length - 1] || "";

        // Remove punctuation and make it lowercase
        lastWord = lastWord.replace(/[^a-zA-Z]/g, "").toLowerCase();

        return lastWord;
    } catch (error) {
        console.error("Error calling the chatCompletion inference", error.response ? error.response.data : error);
        res.status(500).send("Error calling the chatCompletion API");
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
        if (model.type === "chat-completion") {
            llmWord = await chatCompletioncall(model, round, past_words_array, res);
        } else {
            llmWord = await huggingfacecall(model, round, past_words_array, res);
        }

        console.log(`Model ${model.name} returned word: ${llmWord}`);

        // Ensure the function waits for the check before continuing
        let exists = await checkIfWordExists(llmWord);

        // Check if the word has been used before
        let previouslyUsed = checkIfWordPreviouslyUsed(llmWord, past_words_array);

        if (exists && !previouslyUsed) {
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

    // Update DB with the new words
    try {
        let game = await Game.findByPk(gameId);
        let wordsPlayed1 = game.wordsPlayed1 ? JSON.parse(game.wordsPlayed1) : [];
        let wordsPlayed2 = game.wordsPlayed2 ? JSON.parse(game.wordsPlayed2) : [];
        wordsPlayed1.push(newWord);
        wordsPlayed2.push(llmWord);
        await game.update({
            wordsPlayed1: JSON.stringify(wordsPlayed1),
            wordsPlayed2: JSON.stringify(wordsPlayed2),
            roundCount: round,
            status: status
        });
    } catch (err) {
      console.error('Error updating game in DB:', err);
    }

    // Pick a time delay between 1 and 3 seconds
    const delay = Math.floor(Math.random() * 4) + 1;
    // Wait for the delay before sending the response
    setTimeout(() => {
        res.json({llmWord: llmWord, status: status});
    }, delay * 1000);
});

export default router;