const express = require('express');
const path = require("path");
const app = express()
const axios = require('axios');

// Serve all static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "index.html"));
});
app.use(express.json());

// List of available models
const availableModels = [
    { name: 'gpt2', type: 'text-generation', langages : ['en', 'fr'] },
    { name: 'mistralai/Mistral-7B-v0.1', type: 'text-generation', langages : ['en'] },
    { name: 'google/flan-t5-large', type: 'text2text-generation' , langages : ['en'] },
    { name: 'bert-base-uncased', type: 'fill-mask' , langages : ['en'] , mask_token: '[MASK]'},
    { name: 'FacebookAI/xlm-roberta-base', type: 'fill-mask' , langages : ['es', 'en','fr'], mask_token: '<mask>'},
    // Add other models here
];
// Endpoint to get available models
app.get('/available-models', (req, res) => {
    res.json(availableModels);
});

const API_TOKEN = "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";

app.post('/initialize-model', async (req, res) => {
    const model = req.body.model; // Retrieve the model from the request body

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

    try {
        const response = await axios.post(
            `https://api-inference.huggingface.co/models/${model.name}`,
            { inputs: token },
            {
                headers: { Authorization: `Bearer ${API_TOKEN}` }
            }
        );
        res.json(response.data);
    } catch (error) {
        console.error("Error calling the Hugging Face API", error);
        if (error.response && error.response.status === 503) {
            res.status(503).send("Model is loading");
        } else {
            res.status(500).send("Error calling the Hugging Face API");
        }
    }
});

app.post('/query-model', async (req, res) => {
    const past_words_array = req.body.previous_words;
    const model = req.body.model;

    // Check if the model name is valid
    const modelNames = availableModels.map(model => model.name);
    if (!model || typeof model.name !== 'string' || !modelNames.includes(model.name)) {
        return res.status(400).send("Invalid model name");
    }

    // Template for new rounds
    const createRoundTemplate = (roundNumber, pastWords) => {
        return `\nRound ${roundNumber}! Past words, forbidden to use are ${pastWords.join(', ')}. Please give your word for the current round.\n`;
    };

    const RULE_TOKEN = "We are playing a game where at each round we say an word. The goal is to produce the same word based on previous words at which point the game ends. "
    let ROUND_ONE = "Round 1 ! New game, please give your first word.\n"
    let CURRENT_ROUND_COUNT = ROUND_ONE
    if (Array.isArray(past_words_array) && past_words_array.length > 0) {
        let round = Math.floor(past_words_array.length / 2) + 1;
        CURRENT_ROUND_COUNT = createRoundTemplate(round, past_words_array);
    }

    // Example of gameplay
    const EXAMPLES = "\n\nExample of gameplay 1:\n" + 
    RULE_TOKEN + 
    ROUND_ONE +
    "Player 1: 'Apple'\n" +
    "Player 2: 'Banana'\n\n" +
    createRoundTemplate(2, ['Apple','Banana']) +
    "Player 1: 'Fruit'\n" +
    "Player 2: 'Green'\n\n" +
    createRoundTemplate(3, ['Apple','Banana', 'Fruit', 'Green']) +
    "Player 1: 'Vegetable'\n" +
    "Player 2: 'Vegetable'\n\n" + 
    "The game ends as both players said 'Vegetable'.\n\n" +
    "\n\nExample of gameplay 2:\n" + 
    RULE_TOKEN + 
    ROUND_ONE +
    "Player 1: 'House'\n" +
    "Player 2: 'Mountain'\n\n" +
    createRoundTemplate(2, ['House','Mountain']) +
    "Player 1: 'Monastery'\n" +
    "Player 2: 'Cave'\n\n" +
    createRoundTemplate(3, ['House','Mountain', 'Monastery', 'Cave']) +
    "Player 1: 'Secret'\n" +
    "Player 2: 'Monastery'\n\n" + 
    "The game is lost as Player 2 gave a previously given word.\n\n";

    // Construct the rounds history based on past words
    let interactionHistory = RULE_TOKEN + ROUND_ONE
    if (Array.isArray(past_words_array) && past_words_array.length > 0) {
        for (let i = 0; i < past_words_array.length; i++) {
            if (i % 2 === 0) {  // Start a new round every two words
                interactionHistory += createRoundTemplate(Math.floor(i / 2) + 2, past_words_array.slice(0, i));
            interactionHistory += `Player ${i % 2 + 1}: '${past_words_array[i]}'\n`;
            }
        }
    }

    console.log(model.type)
    let token;
    let parameters;
    if (model.type === 'text2text-generation') {
        token = EXAMPLES + interactionHistory + CURRENT_ROUND_COUNT
    }

    if (model.type === 'text-generation') {
        token = EXAMPLES + interactionHistory + CURRENT_ROUND_COUNT + "Player 1 : '"
        parameters = {return_full_text:false, max_new_tokens: 20 }
    }

    if (model.type === 'fill-mask') {
        token = EXAMPLES + interactionHistory + CURRENT_ROUND_COUNT + "Player 1 : '" + model.mask_token +  "'\n."
    }
    try {
        const response = await axios.post(
            `https://api-inference.huggingface.co/models/${model.name}`,
            { inputs: token, parameters: parameters, options: { wait_for_model: true }},
            {
                headers: { Authorization: `Bearer ${API_TOKEN}` },
            }
        );
        console.log(response.data)

        if (model.type === "fill-mask") {
            res.json(response.data[0].token_str);
        }
        else {
            res.json(response.data[0].generated_text.match(/\b\w+\b/)?.[0]);
        } 
    } catch (error) {
        console.error("Error calling the Hugging Face API", error);
        res.status(500).send("Error calling the Hugging Face API");
    }
});


module.exports = app