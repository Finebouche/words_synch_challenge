document.addEventListener('DOMContentLoaded', function () {
    var closeButton = document.getElementById('errorBanner').querySelector('button');
    if (closeButton) {
        closeButton.addEventListener('click', function () {
            document.getElementById('errorBanner').style.display = 'none';
        });
    }
});

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

document.addEventListener('DOMContentLoaded', function () {
    // If there's an errorBanner on the page, set up the close button
    var errorBanner = document.getElementById('errorBanner');
    if (errorBanner) {
        var closeButton = errorBanner.querySelector('button');
        if (closeButton) {
            closeButton.addEventListener('click', function () {
                errorBanner.style.display = 'none';
            });
        }
    }

    // Store references to important elements
    let languageSelect = document.getElementById('languageSelect');
    let selectLLMGame = document.getElementById('selectLLMGame');
    let selectHumanGame = document.getElementById('selectHumanGame');
    let selectionLLM = document.getElementById('selections-LLM');
    let messageHuman = document.getElementById('message-Human');
    let llmSelect = document.getElementById('llmSelect');
    let startGameButton = document.getElementById('startLLMGame');

    // 1) When language is chosen, show/hide the "Play with LLM" and "Play with Human" buttons
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

    // 2) If the user selects "Play with LLM"
    selectLLMGame.addEventListener('click', function () {
        gameMode = 'llm';
        updateModelOptions(selectedLanguage)
        // Show LLM options, hide Human options
        selectionLLM.style.display = 'block';
        messageHuman.style.display = 'none';
        // Hide the "Play with..." buttons after a choice
        selectLLMGame.style.display = 'none';
        selectHumanGame.style.display = 'none';
        languageSelect.disabled = true; // Disable language selection

        // Show the LLM selection dropdown
        llmSelect.style.display = '';

        // Only show the start button if an LLM is selected
        llmSelect.addEventListener('change', function () {
            if (this.value) {
                startGameButton.style.display = '';
            } else {
                startGameButton.style.display = 'none';
            }
        });
    });

    // 3) If the user selects "Play with Human"
    selectHumanGame.addEventListener('click', function () {
        gameMode = 'human';
        socket = io(); // Connect to the server
        // Hide LLM options, and "Play with..." buttons
        selectionLLM.style.display = 'none';
        selectLLMGame.style.display = 'none';
        selectHumanGame.style.display = 'none';
        languageSelect.disabled = true; // Disable language selection
        socket.emit('joinQueue', { language: selectedLanguage, playerId: playerId });

        socket.on('waitingForOpponent', () => {
            // Show a message to the user that we're waiting for an opponent
            messageHuman.style.display = 'block';
        });

        // Suppose we handle "gameStarted" event
        socket.on('gameStarted', ({ gameId: gId, role }) => {
            gameId = gId;
            myRole = role;
            messageHuman.style.display = 'none';
            console.log(`Game started! I am ${myRole} in game ${gameId}.`);
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
let MODELS = []
let selectedModelName = null;
document.addEventListener('DOMContentLoaded', function() {
    fetch('/model/available-models')
        .then(response => response.json())
        .then(availableModels => {
            MODELS = availableModels;
        })
        .catch(error => {
            console.error('Error fetching models:', error);
        });
});

document.getElementById('startLLMGame').addEventListener('click', function () {
    // Get selected model
    selectedModelName = document.getElementById('llmSelect').value;
    // Get the corresponding flag emoji for the selected language
    let languageName = languageNames[selectedLanguage];

    // Update the text of the paragraphs to show the selections
    let selectedModel = MODELS.find(model => model.name === selectedModelName);
    document.getElementById('selectedContent').textContent = "Bzz... bzz... model " + selectedModel.name + " loaded, currently playing with " + languageName  + " vocabulary...";
    document.getElementById('languageSelect').style.display = 'none';

    fetch('/model/initialize-model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model: selectedModel, player_id: playerId, language: selectedLanguage}) // Send the selected model to the server
    })
        .then(response => {
            console.log("Model is loading. Please wait.");
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
    const params = new URLSearchParams({
        action: 'query',
        format: 'json',
        titles: word.toLowerCase(),
        origin: '*'
    });

    const url = `${endpoint}?${params.toString()}`;

    try {
        const response = await fetch(url);
        const data = await response.json();
        if (!data.query.pages['-1']) {
            errorMessageElement.style.display = 'none';
            return true;
        } else {
            errorMessageElement.textContent = getTranslation('errorMessageNotExist');
            errorMessageElement.style.display = 'block';
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
        submitButton.disabled = false;
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

                if (data.status === 'loses') {
                    loose_game();
                } else if (data.status === 'wins') {
                    win_game();
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

        // Once emitted, you can set `haveIPlayedThisRound = true` immediately:
        submitButton.disabled = true;
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

            if (status === 'loses') {
                loose_game();
            } else if (status === 'wins') {
                win_game();
            } else {
                haveIPlayedThisRound = false;
                submitButton.disabled = false
            }
        });
    }

});

document.getElementById('gameRestart').addEventListener('click', async function (event) {
    // Clear previous words and conversation
    document.getElementById('conversationArea').innerHTML = '';
    document.getElementById('previousWordsArea').innerHTML = '';
    document.getElementById('selectedContent').textContent = '';
    document.getElementById('selectedInfo').style.display = 'none';
    document.getElementById('gameWord').value = '';
    document.getElementById('winMessage').style.display = 'none';
    document.getElementById('lossMessage').style.display = 'none';
    past_words_array = [];
    // set selection to default
    document.getElementById('llmSelect').value = '';
    document.getElementById('llmSelect').style.display = 'none';
    document.getElementById('languageSelect').value = '';
    document.getElementById('languageSelect').disabled = false;
    document.getElementById('startLLMGame').style.display = 'none';

    document.getElementById('gameRestart').style.display = 'none';
    document.getElementById('selections-LLM').style.display = 'block';

    // Stop the confettis
    const wrapper = document.getElementById('confetti-wrapper');
    wrapper.innerHTML = '';
    // Stop the rain
    document.querySelectorAll('.rain-wrapper').forEach(function(rainElement) {
        rainElement.innerHTML = '';
    });

});
// END GAME LOGIC
