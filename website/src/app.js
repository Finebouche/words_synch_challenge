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

const API_TOKEN = "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";

app.post('/initialize-model', async (req, res) => {
    const model = req.body.model; // Retrieve the model from the request body

    // Check if the model name is valid
    if (!['mistralai/Mistral-7B-v0.1', 'gpt2'].includes(model)) {
        return res.status(400).send("Invalid model name");
    }

    try {
        const response = await axios.post(
            `https://api-inference.huggingface.co/models/${model}`,
            { inputs: "initialisation" },
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
    const past_words_array = req.body.past_words_array;
    const model = req.body.model; // Retrieve the model from the request body

    const RULE_TOKEN ="You are an ChatBot and we are playing a game where at each round we produce a word. At initial step the word produced is random, but for each following iteration we have to produce the same word. The goal is to produce the same word, at which point the game ends. You will start and keep track of the round and I will then give my word to you and we can pursue to the next round. After round 1, that is the initialization of the game, you will give me your reasoning of why you choose this word, before giving your word for the current round."
    let WORDS = "Please give me your word for the first round."

    if (past_words_array != null) {
        WORDS = `Here are the words that have been played so far: ${past_words_array.join(', ')}. \nPlease give me only the word for the current round.`
    } 

    const token = RULE_TOKEN + WORDS

    try {
        const response = await axios.post(
            `https://api-inference.huggingface.co/models/${model}`,
            { inputs: token },
            {
                headers: { Authorization: `Bearer ${API_TOKEN}` },
            }
        );
        res.json(response.data);
    } catch (error) {
        console.error("Error calling the Hugging Face API", error);
        res.status(500).send("Error calling the Hugging Face API");
    }
});


module.exports = app