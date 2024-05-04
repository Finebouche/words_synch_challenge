// LANGUAGE SELECTION

// Function to get the translation of a key
var translations; // Global variable for translations

async function loadLanguage(lang) {
    const response = await fetch(`locales/${lang}.json`);
    translations = await response.json();

    document.querySelectorAll('[data-translate]').forEach(el => {
        const key = el.getAttribute('data-translate');
        const translation = translations[key];

        if (Array.isArray(translation)) {
            el.innerHTML = ''; // Clear the current list
            if (translation.length > 0) {
                const p = document.createElement('p');
                p.textContent = translation[0];
                el.appendChild(p);
            }

            var ul = document.createElement('ul');
            el.appendChild(ul);
            // Start from the second element (index 1) since the first is already processed
            translation.slice(1).forEach(item => {
                const li = document.createElement('li');
                li.textContent = item;
                ul.appendChild(li);
            });
        } else if (translation) {
            el.textContent = translation;
        }

        if (el.placeholder && translation) {
            el.placeholder = translation; // For input placeholders
        }
    });
}

document.addEventListener('DOMContentLoaded', function () {
    var closeButton = document.getElementById('errorBanner').querySelector('button');
    if (closeButton) {
        closeButton.addEventListener('click', function () {
            document.getElementById('errorBanner').style.display = 'none';
        });
    }
});


document.querySelectorAll('.language-option').forEach(function (element) {
    element.addEventListener('click', function () {
        const selectedLang = this.getAttribute('data-lang');
        loadLanguage(selectedLang);

        // Update current language display
        document.getElementById('currentLanguage').innerHTML = this.innerHTML;
        document.getElementById('languageOptions').style.display = 'none';
    });
});

document.getElementById('currentLanguage').addEventListener('click', function () {
    var languageOptions = document.getElementById('languageOptions');
    languageOptions.style.display = languageOptions.style.display === 'block' ? 'none' : 'block';
});

// Close the language options if clicked outside
document.addEventListener('click', function (event) {
    var languageOptions = document.getElementById('languageOptions');
    var currentLanguage = document.getElementById('currentLanguage');

    // Check if the click is outside the languageOptions and currentLanguage
    if (!currentLanguage.contains(event.target) && !languageOptions.contains(event.target)) {
        languageOptions.style.display = 'none';
    }
});

document.addEventListener('DOMContentLoaded', function () {
    loadLanguage('en'); // Default to English or use browser's language setting
});

function getTranslation(key) {
    if (translations && key in translations) {
        return translations[key];
    }
    return key; // Return the key itself if translation is not found
}

// END LANGUAGE SELECTION


// GAME START
var languageFlags = {
    'en': '🇬🇧', // Flag for English
    'es': '🇪🇸', // Flag for Spanish
    'fr': '🇫🇷', // Flag for French
    // Add more mappings as needed
};
document.addEventListener('DOMContentLoaded', function () {
    var languageSelect = document.getElementById('languageSelect');
    var llmSelect = document.getElementById('llmSelect');
    var startGameButton = document.getElementById('startGame');

    languageSelect.addEventListener('change', function () {
        if (this.value) {
            llmSelect.style.display = '';
        } else {
            llmSelect.style.display = 'none';
            startGameButton.style.display = 'none';
        }
    });

    llmSelect.addEventListener('change', function () {
        if (this.value) {
            startGameButton.style.display = '';
        } else {
            startGameButton.style.display = 'none';
        }
    });
});

var MODELS = []
document.addEventListener('DOMContentLoaded', function() {
    const languageSelect = document.getElementById('languageSelect');
    const modelSelect = document.getElementById('llmSelect');

    fetch('/available-models')
        .then(response => response.json())
        .then(availableModels => {
            MODELS = availableModels;
            updateModelOptions(); // Populate models for the default language
        })
        .catch(error => {
            console.error('Error fetching models:', error);
        });

    languageSelect.addEventListener('change', updateModelOptions);

    function updateModelOptions() {
        const selectedLanguage = languageSelect.value;
        modelSelect.innerHTML = ''; // Clear existing options

        // Add the default "Select a LLM" option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select a LLM';
        defaultOption.setAttribute('data-translate', 'llm-option');
        modelSelect.appendChild(defaultOption);

        // Add options based on the selected language
        MODELS.forEach(model => {
            if (model.langages.includes(selectedLanguage)) {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.name;
                modelSelect.appendChild(option);
            }
        });
    }
});

document.getElementById('startGame').addEventListener('click', function () {
    // Get selected language and model
    var selectedLanguage = document.getElementById('languageSelect').value;
    var selectedModelName = document.getElementById('llmSelect').value;

    // Get the corresponding flag emoji for the selected language
    var flagEmoji = languageFlags[selectedLanguage];

    // Update the text of the paragraphs to show the selections with the flag emoji
    document.getElementById('selectedLanguage').textContent = "Language: " + flagEmoji + " " + selectedLanguage;
    var selectedModel = MODELS.find(model => model.name === selectedModelName);
    document.getElementById('selectedModel').textContent = "Model: " + selectedModel.name + " (" + selectedModel.type + ")";

    // Hide the selections and show the selected information and show the game input
    document.getElementById('selections').style.display = 'none';
    document.getElementById('selectedInfo').style.display = 'block';
    document.getElementById('gameInput').style.display = 'block';

    fetch('/initialize-model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model: selectedModel }) // Send the selected model to the server
    })
        .then(response => {
            console.log("Model is loading. Please wait.");
            if (response.status === 503) {
                console.log("Model is not available. Take another one");
                document.getElementById('errorBanner').style.display = 'block'; // Show the banner
                document.getElementById('selections').style.display = 'block';
                document.getElementById('languageSelect').value = '';
                document.getElementById('llmSelect').value = '';
                document.getElementById('submitWord').disabled = true;
            } else {
                console.log("Model is ready.");
                document.getElementById('submitWord').disabled = false;
            }
        })
        .catch(error => {
            console.error('Error initializing model:', error);
        });
});
// END GAME START

// GAME LOGIC
async function checkWordExistence(word, language) {
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
            return true;
        } else {
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

var past_words_array = []; // Array to store the words

document.getElementById('submitWord').addEventListener('click', async function (event) {
    event.preventDefault(); // Prevent form submission
    var submitButton = document.getElementById('submitWord');
    submitButton.disabled = true; // Disable the submit button
    var word = document.getElementById('gameWord').value.trim();
    var errorMessageElement = document.getElementById('errorMessage');
    var selectedLanguage = document.getElementById('languageSelect').value;

    // check if the input is empty
    if (word === '') {
        errorMessageElement.textContent = getTranslation('errorMessageEmpty');
        errorMessageElement.style.display = 'block';
        return;
    } else {
        errorMessageElement.style.display = 'none';
    }

    // check word existence in the array
    if (past_words_array.includes(word)) {
        errorMessageElement.textContent = getTranslation('errorMessageUsed');
        errorMessageElement.style.display = 'block';
        return;
    }
    else {
        errorMessageElement.style.display = 'none';
    }

    // check in the dictionary
    const wordExists = await checkWordExistence(word, selectedLanguage);
    if (!wordExists) {
        errorMessageElement.textContent = getTranslation('errorMessageNotExist');
        errorMessageElement.style.display = 'block';
        return;
    } else {
        errorMessageElement.style.display = 'none';
    }

    var selectedModelName = document.getElementById('llmSelect').value;
    var selectedModel = MODELS.find(model => model.name === selectedModelName);

    const response = fetch('/query-model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model: selectedModel, previous_words: past_words_array}) // Send the selected model and current words to the server

    })
    .then(response => response.json())
    .then(data => {
        console.log(data)
        let llm_word = data;
        // Add word to array
        past_words_array.push(llm_word);
        past_words_array.push(word);

        // Update conversation area with the latest random word
        document.getElementById('conversationArea').innerHTML += `<div class="bubbleContainer"><div class="message"><span class="emoji">&#x1F60A;</span><span class="bubble left">${word}</span></div><div class="message"><span class="bubble right">${llm_word}</span><span class="emoji">&#x1F916;</span></div></div>`;


        // Clear the input field
        document.getElementById('gameWord').value = '';

        // Update previous words area
        updatePreviousWordsArea();
    })
    .catch(error => {console.error('Error fetching random word:', error);})
    .finally(() => {submitButton.disabled = false;}); // Enable the submit button
    
});
