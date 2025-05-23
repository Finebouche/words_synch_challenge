const languageNames = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    // Add more mappings as needed
};

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) { // `i > 0` ensures every element gets a swap
        const j = Math.floor(Math.random() * (i + 1)); // Random index from 0 to i
        [array[i], array[j]] = [array[j], array[i]]; // Swap elements
    }
    return array;
}

///////////////////////////
///// GAME SELECTION //////
///////////////////////////

let selectedLanguage = null;
let past_words_array = []; // Array to store the words
let socket = null;
let gameId = null;
let gameMode = null;
let shownMode = null;
let myRole = null;
let MODELS = []
let selectedModelName = null;

const NUMBERS_OF_GAME_PER_CONFIG = 4;
const CAN_SELECT_LLM = true;

// based on gameConfigOrder, gamesCount and NUMBERS_OF_GAME_PER_CONFIG determines the next gameConfig
function getNextGameConfig(gameConfigOrder, gamesCount) {
    let nextGameConfig = gameConfigOrder[0];
    for (let i = 0; i < gameConfigOrder.length; i++) {
        nextGameConfig = gameConfigOrder[i];
        if (gamesCount[gameConfigOrder[i]] < NUMBERS_OF_GAME_PER_CONFIG) {
            break;
        }
        else if (i === gameConfigOrder.length - 1) {
            nextGameConfig = "all_games_played";
        }
    }
    return nextGameConfig
}

/**
 * These elements are used later for UI interactions.
 */
let selectLLMGame = document.getElementById('selectLLMGame'); // Button for "Play with LLM"
let selectHumanGame = document.getElementById('selectHumanGame'); // Button for "Play with Human"
let selectionLLM = document.getElementById('selections-LLM'); // LLM game settings container
let llmSelect = document.getElementById('llmSelect'); // LLM selection dropdown
let startGameButton = document.getElementById('startLLMGame'); // Button to start LLM game
let languageSelect = document.getElementById('languageSelect'); // Language selection dropdown
let messageHuman = document.getElementById('message-Human'); // Message shown while waiting for a human opponent
let messageLLM = document.getElementById('message-LLM'); // Message shown while waiting for a LLM opponent
let llmSelectedContent = document.getElementById('selectedContent'); // Message shown after selecting a LLM model

function initialiseHumanGame(gameConfig) {
    /*
    * - nextGameConfig can be either 'human_vs_human_(bot_shown)' or 'human_vs_human_(human_shown)'
    */

    let playerId = localStorage.getItem('connectedPlayerId') || localStorage.getItem('newPlayerId');
    let languageName = languageNames[selectedLanguage];
    let playerGameConfigOrder = JSON.parse(localStorage.getItem('gameConfigOrder'));

    gameMode = 'human';

    // Hide game selection and LLM-related UI elements
    selectionLLM.style.display = 'none';
    selectLLMGame.style.display = 'none';
    selectHumanGame.style.display = 'none';
    languageSelect.style.display = 'none';

    if (gameConfig === 'human_vs_human_(bot_shown)') {
        messageLLM.style.display = 'block';
        shownMode = 'bot';
    } else {
        messageHuman.style.display = 'block';
        shownMode = 'human';
    }
    // Join the matchmaking queue for a human game
    socket.emit('joinQueue', { language: selectedLanguage, playerId: playerId, gameConfig, gameConfigOrder: playerGameConfigOrder });

    // Handle matchmaking status updates
    socket.off('waitingForOpponent'); // Remove previous listeners
    socket.on('waitingForOpponent', () => {});

    // Handle game start event from the server
    socket.off('gameStarted'); // Remove previous listeners
    socket.on('gameStarted', ({ gameId: gId, role }) => {
        gameId = gId; // Store game ID
        myRole = role; // Store player role

        // Hide waiting messages
        messageHuman.style.display = 'none';
        messageLLM.style.display = 'none';
        if (gameConfig === 'human_vs_human_(bot_shown)') {
            document.getElementById('selectedInfo').style.display = 'block';
            llmSelectedContent.style.display = 'block';
            selectedModelName = 'gpt-4o';
            llmSelectedContent.textContent = "Bzz... bzz... model " + selectedModelName + " loaded, currently playing with " + languageName  + " vocabulary...";
        }

        console.log(`Game started! I am ${myRole} in game ${gameId}.`);

        // Show game input UI
        document.getElementById('gameInput').style.display = 'block';
        document.getElementById('submitWord').disabled = false;
    });
}



function initialiseBotGame(gameConfig) {
    /**
    * - Updates the game mode to 'llm'
    */
    gameMode = 'llm';

    if (CAN_SELECT_LLM) {
        updateModelOptions(selectedLanguage); // Load available LLMs for the selected language

        // Show LLM-related UI elements
        selectionLLM.style.display = 'block';
        llmSelect.style.display = '';
        llmSelect.addEventListener('change', function () {
            startGameButton.style.display = this.value ? '' : 'none';
        });
        document.getElementById('startLLMGame').addEventListener('click', function () {
            // Get selected model
            selectedModelName = document.getElementById('llmSelect').value;
            loadModelAndStartGame(selectedModelName, "human_vs_bot_(bot_shown)");
        });
    } else {
        // Hide game mode selection buttons and language dropdown
        selectLLMGame.style.display = 'none';
        selectHumanGame.style.display = 'none';
        languageSelect.style.display = 'none';
        selectedModelName = 'gpt-4o';
        loadModelAndStartGame(selectedModelName, gameConfig);
    }
}


function loadModelAndStartGame(model_name, gameConfig) {

    let playerId = localStorage.getItem('connectedPlayerId') || localStorage.getItem('newPlayerId');
    let languageName = languageNames[selectedLanguage];
    let selectedModel = MODELS.find(model => model.name === model_name);
    let playerGameConfigOrder = JSON.parse(localStorage.getItem('gameConfigOrder'));

    document.getElementById('languageSelect').style.display = 'none';
    // gameConfig is 'human_vs_bot_(bot_shown)' or 'human_vs_bot_(human_shown)'
    if (gameConfig === 'human_vs_bot_(bot_shown)') {
        messageLLM.style.display = 'block';
        shownMode = 'bot';
    } else {
        messageHuman.style.display = 'block';
        shownMode = 'human';
    }
    console.log("Model is loading. Please wait.");

    fetch('/model/initialize-model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ model: selectedModel, player_id: playerId, language: selectedLanguage, game_config: gameConfig, game_config_order: playerGameConfigOrder}) // Send the selected model to the server
    })
        .then(response => {
            document.getElementById('message-LLM').style.display = 'none';
            if (!response.ok) {
                console.log("Model is not available. Take another one");
                document.getElementById('errorBanner').style.display = 'block'; // Show the banner
                llmSelect.value = '';
                document.getElementById('submitWord').disabled = true;
                document.getElementById('startLLMGame').style.display = 'none';
                throw new Error(`Server returned status ${response.status}`);
            } else {
                console.log("Model is ready.");
                document.getElementById('selections-LLM').style.display = 'none';
                document.getElementById('gameInput').style.display = 'block';
                document.getElementById('submitWord').disabled = false;
                return response.json()
            }

        })
        .then(data => {
            gameId = data.gameId;
            messageLLM.style.display = 'none';
            messageHuman.style.display = 'none';
            console.log("gameConfig", gameConfig);
            if (gameConfig === 'human_vs_bot_(bot_shown)') {
                document.getElementById('selectedInfo').style.display = 'block';
                llmSelectedContent.style.display = 'block';
                llmSelectedContent.textContent = "Bzz... bzz... model " + selectedModelName + " loaded, currently playing with " + languageName  + " vocabulary...";
            }
        })
        .catch(error => {
            console.error('Error initializing model:', error);
        });
}

function initialiseGameSetup() {
    let playerId = localStorage.getItem('connectedPlayerId') || localStorage.getItem('newPlayerId');
    let nextGameConfig;

    /**
     * 4)  Initialize WebSocket connection
     */
    socket = io();

    fetch(`/auth/exists`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ playerId: playerId })
    })
    .then(response => response.json()) // Convert response to JSON
    .then(data => {
        if (!data.exists) {
             // that means that this player never played any game before
            // we keep gameConfigOrder random and gamesCount to 0 as before
            return Promise.resolve();
        } else {
            // If the player exists, fetch the games configuration count from the database.
            return fetch(`/auth/games-config-count`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ playerId: playerId })
                })
                .then(response => response.json())
                .then(configData => {
                    // Assign the fetched data to our variables.
                    localStorage.setItem('gamesCount', JSON.stringify(configData.gamesCount));
                    localStorage.setItem('gameConfigOrder', JSON.stringify(configData.gameConfigOrder));
                });
        }
    }).then(() => {
        console.log('Game configuration order:', JSON.parse(localStorage.getItem('gameConfigOrder')));
        console.log('Games count:', JSON.parse(localStorage.getItem('gamesCount')));

        // nextGameConfig = getNextGameConfig(
        //     JSON.parse(localStorage.getItem('gameConfigOrder')),
        //     JSON.parse(localStorage.getItem('gamesCount')),
        // )
        nextGameConfig = "no_config"; // for display
        console.log('Next game configuration:', nextGameConfig);

        /**
         * 5) Depending on the next game configuration, show the corresponding game mode selection button
         * and handle the corresponding game mode selection
         */
        if (nextGameConfig === 'human_vs_human_(bot_shown)' || nextGameConfig === 'human_vs_bot_(bot_shown)') {
            selectLLMGame.style.display = 'block';
            selectHumanGame.style.display = 'none';
            if (nextGameConfig === 'human_vs_human_(bot_shown)') {
                selectLLMGame.onclick = function () {
                    initialiseHumanGame(nextGameConfig)
                };
            }
            else if (nextGameConfig === 'human_vs_bot_(bot_shown)') {
                selectLLMGame.onclick = function () {
                    initialiseBotGame(nextGameConfig)
                };
            }
        }
        else if (nextGameConfig === 'human_vs_human_(human_shown)' || nextGameConfig === 'human_vs_bot_(human_shown)') {
            selectHumanGame.style.display = 'block';
            selectLLMGame.style.display = 'none';
            if (nextGameConfig === 'human_vs_human_(human_shown)') {
                selectHumanGame.onclick = function () {
                    initialiseHumanGame(nextGameConfig)
                };
            }
            else if (nextGameConfig === 'human_vs_bot_(human_shown)') {
                selectHumanGame.onclick = function () {
                    initialiseBotGame(nextGameConfig)
                };
            }
        } else if (nextGameConfig === 'all_games_played') {
            console.log('All games have been played');
            document.getElementById('returnToProlific').style.display = 'block';
        }
        else if (nextGameConfig === 'no_config') {
            // Display both buttons with no deception
            selectLLMGame.style.display = 'block';
            selectHumanGame.style.display = 'block';
            selectLLMGame.onclick = function () {
                initialiseBotGame("human_vs_bot_(bot_shown)")
            };
            selectHumanGame.onclick = function () {
                initialiseHumanGame("human_vs_human_(human_shown)")
            };
        }
    })
}

document.addEventListener('DOMContentLoaded', function () {
    console.log('newPlayerId', localStorage.getItem('newPlayerId'));
    /**
     * 1) Setup Error Banner Handling
     * If an error banner exists on the page, this ensures that clicking the close button hides it.
     */
    let errorBanner = document.getElementById('errorBanner');
    if (errorBanner) {
        let closeButton = errorBanner.querySelector('button');
        if (closeButton) {
            closeButton.addEventListener('click', function () {
                errorBanner.style.display = 'none';
            });
        }
    }
    /**
     * 2) Handle Language Selection
     * - If the language selection dropdown is disabled, default to English and show game mode buttons.
     * - Otherwise, listen for user selection and show/hide game mode buttons accordingly.
     */
    if (languageSelect.disabled) {
        // selection is disabled -> Default to English if
        selectLLMGame.style.display = 'block';
        selectHumanGame.style.display = 'block';
        languageSelect.style.display = 'none';
        selectedLanguage = 'en';
    } else {
        // Listen for language selection changes
        languageSelect.onchange = function () {
            if (this.value) {
                selectLLMGame.style.display = 'block';
                selectHumanGame.style.display = 'block';
                selectedLanguage = this.value;
            } else {
                selectLLMGame.style.display = 'none';
                selectHumanGame.style.display = 'none';
            }
        };
    }

    /**
     * 3) Fetch Available Models
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
     * 5) Initialize game Configuration variables
     *
     */
    let gameConfigOrder = ['human_vs_human_(bot_shown)', 'human_vs_bot_(bot_shown)', 'human_vs_human_(human_shown)', 'human_vs_bot_(human_shown)']
    let gamesCount = {
        'human_vs_human_(bot_shown)': 0,
        'human_vs_bot_(bot_shown)': 0,
        'human_vs_human_(human_shown)': 0,
        'human_vs_bot_(human_shown)': 0
    }
    // mix the order randomly
    gameConfigOrder = shuffleArray(gameConfigOrder)
    localStorage.setItem('gameConfigOrder', JSON.stringify(gameConfigOrder));
    localStorage.setItem('gamesCount', JSON.stringify(gamesCount));

    initialiseGameSetup();


    /**
     * 6) Optionnally, display if a player is waiting in the lobby
     */

    // socket.on('lobbyCountUpdate', (count) => {
    //   const waitingBadge = document.getElementById('waitingBadge');
    //   if (!waitingBadge) return;
    //
    //   if (count > 0) {
    //     // Show the badge
    //     waitingBadge.style.display = 'block';
    //   } else {
    //     // Hide the badge
    //     waitingBadge.style.display = 'none';
    //   }
    // });

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

    // Group models by provider and add options to the dropdown
    const groupedModels = MODELS.reduce((acc, model) => {
        if (model.languages.includes(selected_language)) {
            (acc[model.provider] ||= []).push(model);
        }
        return acc;
    }, {});


    // Add options based on the selected language
    Object.entries(groupedModels).forEach(([provider, models]) => {
        const optGroup = Object.assign(document.createElement('optgroup'), { label: provider });
        models.forEach(({ name, disabled }) => {
            optGroup.appendChild(Object.assign(document.createElement('option'), { value: name, textContent: name, disabled }));
        });
        modelSelect.appendChild(optGroup);
    });
}




///////////////////////////
/////// GAME LOGIC ////////
///////////////////////////

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

    // check word existence in the array with both first letter capitalized and all lowercase
    if (past_words_array.includes(word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()) || past_words_array.includes(word.toLowerCase())) {
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
    let previousWords = past_words_array.slice();
    document.getElementById('previousWordsArea').innerHTML = previousWords.join(', ');
}

function scrollToBottom() {
    const chatContainer = document.getElementById('conversationArea');
    chatContainer.scrollTop = chatContainer.scrollHeight;
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
    let emoji
    if (shownMode === 'bot') {
        emoji = '&#x1F916;';
    } else if (shownMode === 'human') {
        emoji = '&#x1F60A;';
    }
    else {
        console.error('Unknown mode:', shownMode);
    }
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
                document.getElementById('conversationArea').innerHTML += `<div class="bubbleContainer"><div class="message"><span class="emoji">&#x1F60A;</span><span class="bubble left">${word}</span></div><div class="message"><span class="bubble right">${llm_word}</span><span class="emoji">${emoji}</span></div></div>`;
                document.getElementById('gameWord').value = ''; // Clear the input field
                updatePreviousWordsArea(); // Update the list of previous words
                scrollToBottom();

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
                  <span class="emoji">${emoji}</span>
                </div>
              </div>
            `;
            document.getElementById('gameWord').value = ''; // Clear the input field

            // Add both words to the past_words_array
            past_words_array.push(yourWord);
            past_words_array.push(opponentWord);
            updatePreviousWordsArea();
            scrollToBottom();

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

cleanPreviousGameArea = function() {
    document.getElementById('conversationArea').innerHTML = '';
    document.getElementById('selectedContent').textContent = '';
    document.getElementById('selectedInfo').style.display = 'none';
    document.getElementById('gameWord').value = '';
    document.getElementById('winMessage').style.display = 'none';
    document.getElementById('lossMessage').style.display = 'none';
    document.getElementById('gameRestart').style.display = 'none';
}

resetTheGame = function() {
    if (socket) {
        socket.disconnect(); // Disconnect any existing socket connection
        socket = null;
    }
    selectedLanguage = null;
    past_words_array = []; // Array to store the words
    socket = null;
    gameId = null;
    gameMode = null;
    myRole = null;

    // set selection to default
    document.getElementById('previousWordsArea').innerHTML = '';
    let languageSelect = document.getElementById('languageSelect');
    if (languageSelect.disabled) {
        // document.getElementById('selectLLMGame').style.display = 'block';
        // document.getElementById('selectHumanGame').style.display = 'block';
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

    // Number of games
    let playerId = localStorage.getItem('connectedPlayerId') || localStorage.getItem('newPlayerId');
    if (playerId) {
        fetchGameStats();
    }
}

document.getElementById('restartButton').addEventListener('click', async function (event) {
    cleanPreviousGameArea()
    resetTheGame();
    initialiseGameSetup()
});
// END GAME LOGIC
