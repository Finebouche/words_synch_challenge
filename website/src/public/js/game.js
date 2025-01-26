var languageNames = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    // Add more mappings as needed
};
let selectedLanguage = null;
let past_words_array = []; // Array to store the words
let socket = null;
let gameId = null;
let gameMode = null;
let myRole = null;
let MODELS = []
let selectedModelName = null;

document.addEventListener('DOMContentLoaded', function () {
    /**
     * 1️⃣ Setup Error Banner Handling
     * If an error banner exists on the page, this ensures that clicking the close button hides it.
     */
    const errorBanner = document.getElementById('errorBanner');
    if (errorBanner) {
        const closeButton = errorBanner.querySelector('button');
        if (closeButton) {
            closeButton.addEventListener('click', function () {
                errorBanner.style.display = 'none';
            });
        }
    }

    /**
     * 2️⃣ Fetch Available Models
     * Retrieves a list of available LLM models from the server and stores them in the global `MODELS` array.
     */
    fetch('/model/available-models')
        .then(response => response.json())
        .then(availableModels => {
            MODELS = availableModels; // Store fetched models globally
        })
        .catch(error => {
            console.error('Error fetching models:', error);
        });

    /**
     * 3️⃣ Store References to Important UI Elements
     * These elements are used later for UI interactions.
     */
    let selectLLMGame = document.getElementById('selectLLMGame'); // Button for "Play with LLM"
    let selectHumanGame = document.getElementById('selectHumanGame'); // Button for "Play with Human"
    let selectionLLM = document.getElementById('selections-LLM'); // LLM game settings container
    let messageHuman = document.getElementById('message-Human'); // Message shown while waiting for a human opponent
    let llmSelect = document.getElementById('llmSelect'); // LLM selection dropdown
    let startGameButton = document.getElementById('startLLMGame'); // Button to start LLM game

    let languageSelect = document.getElementById('languageSelect'); // Language selection dropdown

    /**
     * 4️⃣ Handle Language Selection
     * - If the language selection dropdown is disabled, default to English and show game mode buttons.
     * - Otherwise, listen for user selection and show/hide game mode buttons accordingly.
     */
    if (languageSelect.disabled) {
        // Default to English if selection is disabled
        selectLLMGame.style.display = 'block';
        selectHumanGame.style.display = 'block';
        languageSelect.style.display = 'none';
        selectedLanguage = 'en';
    } else {
        // Listen for language selection changes
        languageSelect.addEventListener('change', function () {
            if (this.value) {
                selectLLMGame.style.display = 'block';
                selectHumanGame.style.display = 'block';
                selectedLanguage = this.value;
            } else {
                selectLLMGame.style.display = 'none';
                selectHumanGame.style.display = 'none';
            }
        });
    }

    /**
     * 5️⃣ Handle "Play with LLM" Selection
     * - Updates the game mode to 'llm'
     * - Displays relevant LLM game settings
     * - Hides unnecessary UI elements
     * - Listens for LLM selection before enabling the "Start Game" button
     */
    selectLLMGame.addEventListener('click', function () {
        gameMode = 'llm';
        updateModelOptions(selectedLanguage); // Load available LLMs for the selected language

        // Show LLM-related UI elements
        selectionLLM.style.display = 'block';
        llmSelect.style.display = '';
        messageHuman.style.display = 'none';

        // Hide game mode selection buttons and language dropdown
        selectLLMGame.style.display = 'none';
        selectHumanGame.style.display = 'none';
        languageSelect.style.display = 'none';

        // Enable start button only when an LLM is selected
        llmSelect.addEventListener('change', function () {
            startGameButton.style.display = this.value ? '' : 'none';
        });
    });

    /**
     * 6️⃣ Handle "Play with Human" Selection
     * - Updates game mode to 'human'
     * - Establishes a WebSocket connection
     * - Sends a request to join the matchmaking queue
     * - Handles various game-related WebSocket events
     */
    selectHumanGame.addEventListener('click', function () {
        let playerId = localStorage.getItem('playerId') || getLocalStorageValue('newPlayerID');
        gameMode = 'human';
        socket = io(); // Initialize WebSocket connection

        // Hide game selection and LLM-related UI elements
        selectionLLM.style.display = 'none';
        selectLLMGame.style.display = 'none';
        selectHumanGame.style.display = 'none';
        languageSelect.style.display = 'none';

        // Join the matchmaking queue for a human game
        socket.emit('joinQueue', { language: selectedLanguage, playerId: playerId });

        // Handle matchmaking status updates
        socket.on('waitingForOpponent', () => {
            // Inform the player that they are waiting for an opponent
            messageHuman.style.display = 'block';
        });

        // Handle game start event from the server
        socket.on('gameStarted', ({ gameId: gId, role }) => {
            gameId = gId; // Store game ID
            myRole = role; // Store player role

            messageHuman.style.display = 'none'; // Hide waiting message
            console.log(`Game started! I am ${myRole} in game ${gameId}.`);

            // Show game input UI
            document.getElementById('gameInput').style.display = 'block';
            document.getElementById('submitWord').disabled = false;
        });
    });
});

function updateModelOptions(selected_language) {
    const modelSelect = document.getElementById('llmSelect');

    modelSelect.innerHTML = ''; // Clear existing options

    // Add the default "Select a LLM" option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select a LLM';
    defaultOption.setAttribute('data-translate', 'llm-option');
    modelSelect.appendChild(defaultOption);

    // Add options based on the selected language
    MODELS.forEach(model => {
        if (model.languages.includes(selected_language)) {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.name;
            modelSelect.appendChild(option);
        }
    });
}

// GAME LOGIC FOR LLMs
document.getElementById('startLLMGame').addEventListener('click', function () {
    let playerId = localStorage.getItem('playerId') || getLocalStorageValue('newPlayerID');

    // Get selected model
    selectedModelName = document.getElementById('llmSelect').value;
    // Get the corresponding flag emoji for the selected language
    let languageName = languageNames[selectedLanguage];

    // Update the text of the paragraphs to show the selections
    let selectedModel = MODELS.find(model => model.name === selectedModelName);
    document.getElementById('selectedContent').textContent = "Bzz... bzz... model " + selectedModel.name + " loaded, currently playing with " + languageName  + " vocabulary...";
    document.getElementById('languageSelect').style.display = 'none';
    document.getElementById('message-LLM').style.display = 'block';
    console.log("Model is loading. Please wait.");

    fetch('/model/initialize-model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model: selectedModel, player_id: playerId, language: selectedLanguage}) // Send the selected model to the server
    })
        .then(response => {
            document.getElementById('message-LLM').style.display = 'none';
            if (response.status === 504 || response.status === 503 || response.status === 500) {
                console.log("Model is not available. Take another one");
                document.getElementById('errorBanner').style.display = 'block'; // Show the banner
                document.getElementById('llmSelect').value = '';
                document.getElementById('submitWord').disabled = true;
                document.getElementById('startLLMGame').style.display = 'none';
                return null; // Stop further processing
            } else {
                console.log("Model is ready.");
                document.getElementById('selections-LLM').style.display = 'none';
                document.getElementById('selectedInfo').style.display = 'block';
                document.getElementById('gameInput').style.display = 'block';
                document.getElementById('submitWord').disabled = false;
                return response.json()
            }

        })
        .then(data => {
            gameId = data.gameId;
        })
        .catch(error => {
            console.error('Error initializing model:', error);
        });
});

async function checkWord(word, language) {
    let errorMessageElement = document.getElementById('errorMessage');
    let submitButton = document.getElementById('submitWord');
    submitButton.disabled = true

    // check if the input is empty
    if (word === '') {
        errorMessageElement.textContent = getTranslation('errorMessageEmpty');
        errorMessageElement.style.display = 'block';
        submitButton.disabled = false;
        return;
    } else {
        errorMessageElement.style.display = 'none';
    }

    // check word existence in the array
    if (past_words_array.includes(word)) {
        errorMessageElement.textContent = getTranslation('errorMessageUsed');
        errorMessageElement.style.display = 'block';
        submitButton.disabled = false;
        return;
    }
    else {
        errorMessageElement.style.display = 'none';
    }

    const endpoint = `https://${language}.wiktionary.org/w/api.php`;

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

    try {
        // Fetch with the word entirely in lowercase
        const lowerCaseData = await fetchWordInfo(word.toLowerCase());
        // Fetch with only the first letter capitalized
        const firstCapData = await fetchWordInfo(word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());

        // Check for valid pages in response data
        if (!lowerCaseData.query.pages['-1'] || !firstCapData.query.pages['-1']) {
            errorMessageElement.style.display = 'none';
            return true;
        } else {
            errorMessageElement.textContent = getTranslation('errorMessageNotExist');
            errorMessageElement.style.display = 'block';
            submitButton.disabled = false;
            return false;
        }
    } catch (error) {
        console.error('Error fetching data from Wiktionary:', error);
        return false; // Return false in case of an error
    }
}

function updatePreviousWordsArea() {
    let previousWordsSorted = past_words_array.slice().sort();
    document.getElementById('previousWordsArea').innerHTML = previousWordsSorted.join(', ');
}

document.getElementById('submitWord').addEventListener('click', async function (event) {
    event.preventDefault(); // Prevent form submission
    let submitButton = document.getElementById('submitWord');
    submitButton.disabled = true; // Disable the submit button

    let word = document.getElementById('gameWord').value.trim();

    // check in the dictionary
    const wordExists = await checkWord(word, selectedLanguage);
    if (!wordExists) {
        return;
    }

    let selectedModel = MODELS.find(model => model.name === selectedModelName);

    if (gameMode === 'llm') {
        const response = fetch('/model/query-model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                model: selectedModel,
                game_id: gameId,
                previous_words: past_words_array,
                new_word: word
            })
        })
            .then(response => response.json())
            .then(data => {
                console.log(data)
                let llm_word = data.llmWord;
                // Add word to array
                past_words_array.push(llm_word);
                past_words_array.push(word);
                document.getElementById('conversationArea').innerHTML += `<div class="bubbleContainer"><div class="message"><span class="emoji">&#x1F60A;</span><span class="bubble left">${word}</span></div><div class="message"><span class="bubble right">${llm_word}</span><span class="emoji">&#x1F916;</span></div></div>`;
                document.getElementById('gameWord').value = ''; // Clear the input field
                updatePreviousWordsArea(); // Update the list of previous words

                if (data.status === 'lost') {
                    loseGame();
                } else if (data.status === 'won') {
                    winGame();
                }
            })
            .catch(error => {
                console.error('Error fetching random word:', error);
            })
            .finally(() => {
                submitButton.disabled = false;
            }); // Enable the submit button
    } else if (gameMode === 'human') {
        // =============== HUMAN VS HUMAN MODE ===============

        // Send to server
        console.log('Submitting word:', word, 'for game', gameId, 'as', myRole);
        socket.emit('submitWordHuman', {
          gameId,
          word,
          role: myRole
        })

        socket.off('roundResult');

        // When the server says "roundResult", we display the word
        socket.on('roundResult', ({ yourWord, opponentWord, status }) => {
            // Display the user’s word (yourWord) and the opponent’s word (opponentWord).
            document.getElementById('conversationArea').innerHTML += `
              <div class="bubbleContainer">
                <div class="message">
                  <span class="emoji">&#x1F60A;</span>
                  <span class="bubble left">${yourWord}</span>
                </div>
                <div class="message">
                  <span class="bubble right">${opponentWord}</span>
                  <span class="emoji">&#x1F60A;</span>
                </div>
              </div>
            `;
            document.getElementById('gameWord').value = ''; // Clear the input field

            // Add both words to the past_words_array
            past_words_array.push(yourWord);
            past_words_array.push(opponentWord);
            updatePreviousWordsArea();

            console.log('Game status:', status);

            if (status === 'lost') {
                loseGame();
            } else if (status === 'won') {
                winGame();
            } else {
                submitButton.disabled = false
            }
        });
    }

});

cleanPreviousWordsArea = function() {
    document.getElementById('conversationArea').innerHTML = '';
    document.getElementById('previousWordsArea').innerHTML = '';
    document.getElementById('selectedContent').textContent = '';
    document.getElementById('selectedInfo').style.display = 'none';
    document.getElementById('gameWord').value = '';
    document.getElementById('winMessage').style.display = 'none';
    document.getElementById('lossMessage').style.display = 'none';
    document.getElementById('gameRestart').style.display = 'none';
}

resetTheGame = function() {
    selectedLanguage = null;
    past_words_array = []; // Array to store the words
    socket = null;
    gameId = null;
    gameMode = null;
    myRole = null;

    // set selection to default
    let languageSelect = document.getElementById('languageSelect');
    if (languageSelect.disabled) {
        document.getElementById('selectLLMGame').style.display = 'block';
        document.getElementById('selectHumanGame').style.display = 'block';
        languageSelect.style.display = 'none';
        selectedLanguage = 'en';
    }
    else {
        languageSelect.value = '';
        document.getElementById('llmSelect').value = '';
        document.getElementById('llmSelect').style.display = 'none';
        document.getElementById('startLLMGame').style.display = 'none';
        document.getElementById('selections-LLM').style.display = 'block';
    }

    // Stop the confettis
    const wrapper = document.getElementById('confetti-wrapper');
    wrapper.innerHTML = '';
    // Stop the rain
    document.querySelectorAll('.rain-wrapper').forEach(function(rainElement) {
        rainElement.innerHTML = '';
    });
}
document.getElementById('restartButton').addEventListener('click', async function (event) {
    cleanPreviousWordsArea()
    resetTheGame();
    fetchGameStats();
});
// END GAME LOGIC

// START QUESTIONNAIRE LOGIC
document.getElementById('questionsButton').addEventListener('click', function() {
    document.getElementById('questionnaireContainer').style.display = 'block';

    // Clear previous words and conversation area
    cleanPreviousWordsArea()
});

// 2. Submit the questionnaire
document.getElementById('submitQuestionnaire').addEventListener('click', function() {
    const strategyUsed = document.getElementById('strategyUsed').value.trim();
    const otherPlayerStrategy = document.getElementById('otherPlayerStrategy').value.trim();

    fetch('/game/answers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            gameId: gameId,
            strategyUsed: strategyUsed,
            otherPlayerStrategy: otherPlayerStrategy
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Questionnaire answers saved successfully!', data);
            // Hide the questionnaire (optional)
            document.getElementById('questionnaireContainer').style.display = 'none';
            // Clear the inputs
            document.getElementById('strategyUsed').value = '';
            document.getElementById('otherPlayerStrategy').value = '';
        } else {
            console.error('Failed to save questionnaire answers.', data);
        }
    })
    .catch(err => {
        console.error('Error saving questionnaire answers:', err);
    });

    document.getElementById('questionnaireContainer').style.display = 'none';
    document.getElementById('thankYouContainer').style.display = 'block';
});

document.getElementById('restartButtonTwo').addEventListener('click', async function (event) {
    document.getElementById('thankYouContainer').style.display = 'none';

    resetTheGame();
});
// END QUESTIONNAIRE LOGIC
