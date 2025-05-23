// START QUESTIONNAIRE LOGIC
document.getElementById('questionsButton').addEventListener('click', function() {
    document.getElementById('questionnaireContainer').style.display = 'block';
    cleanPreviousGameArea();
});

function limitCheckboxSelection(groupName, maxSelection) {
    const checkboxes = document.querySelectorAll(`input[name="${groupName}"]`);
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function () {
            const checkedBoxes = Array.from(checkboxes).filter(cb => cb.checked);
            if (checkedBoxes.length > maxSelection) {
                // Uncheck the first selected checkbox if maximum exceeded
                checkedBoxes[0].checked = false;
            }
        });
    });
}

// Call the function to limit selections to 3 per group
document.addEventListener('DOMContentLoaded', function () {
    limitCheckboxSelection('qualitativeStrategyUsed', 3);
    limitCheckboxSelection('qualitativeOtherPlayerStrategy', 3);
});

function displayErrorMessage(message) {
    const errorDiv = document.getElementById('errorMessageQuestionnaire');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

document.getElementById('submitQuestionnaire').addEventListener('click', function() {
    // === VALIDATION SECTION ===

    // Retrieve radio buttons / checkboxes selections
    const quantitativeStrategyUsed = Array.from(document.querySelector('input[name="quantitativeStrategyUsed"]:checked'));
    const qualitativeStrategyUsed = Array.from(document.querySelectorAll('input[name="qualitativeStrategyUsed"]:checked'));
    const quantitativeOtherPlayerStrategy = Array.from(document.querySelectorAll('input[name="quantitativeOtherPlayerStrategy"]:checked'));
    const qualitativeOtherPlayerStrategy = Array.from(document.querySelectorAll('input[name="qualitativeOtherPlayerStrategy"]:checked'));
    
    // Retrieve ratings (ensure your HTML elements have these unique IDs)
    const otherPlayerUnderstood = document.getElementById('otherPlayerUnderstoodYourStrategies')?.value || '';
    const didYouUnderstandOtherPlayerStrategy = document.getElementById('didYouUnderstandOtherPlayerStrategy')?.value || '';
    const otherPlayerRating = document.getElementById('otherPlayerRating')?.value || '';
    const connectionFeeling = document.getElementById('connectionFeeling')?.value || '';

    // Validate required fields; if any is missing, log a message and abort submission.
    // Validate required fields; if any is missing, update the error message and abort submission.
    if (!quantitativeStrategyUsed) {
        displayErrorMessage('Please answer how you reacted to other player word.');
        return;
    }
    if (qualitativeStrategyUsed.length === 0) {
        displayErrorMessage('Please select at least one qualitative strategy used.');
        return;
    }
    if (quantitativeOtherPlayerStrategy.length === 0) {
        displayErrorMessage('Please answer how the other player reacted to your word.');
        return;
    }
    if (qualitativeOtherPlayerStrategy.length === 0) {
        displayErrorMessage('Please select at least one qualitative other player strategy.');
        return;
    }
    if (!otherPlayerUnderstood) {
        displayErrorMessage('Please provide your rating for how well the other player understood your strategies.');
        return;
    }
    if (!didYouUnderstandOtherPlayerStrategy) {
        displayErrorMessage('Please provide your rating for how well you understood the other player’s strategy.');
        return;
    }
    if (!otherPlayerRating) {
        displayErrorMessage('Please provide your rating for the other player.');
        return;
    }
    if (!connectionFeeling) {
        displayErrorMessage('Please provide your connection feeling rating.');
        return;
    }

    let playerId = localStorage.getItem('connectedPlayerId') || getLocalStorageValue('newPlayerId');

    console.log('Player ID for questionnaire:', playerId);

    // === SUBMISSION SECTION ===
    // All fields are filled; proceed with sending the data to the server.
    fetch('/game/answers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            gameId: gameId,
            role: myRole,
            playerId: playerId,
            quantitativeStrategyUsed: quantitativeStrategyUsed.map(cb => cb.value),
            quantitativeOtherPlayerStrategy: quantitativeOtherPlayerStrategy.map(cb => cb.value),
            qualitativeStrategyUsed: qualitativeStrategyUsed.map(cb => cb.value),
            qualitativeOtherPlayerStrategy: qualitativeOtherPlayerStrategy.map(cb => cb.value),
            otherPlayerUnderstoodYourStrategies: otherPlayerUnderstood,
            didYouUnderstandOtherPlayerStrategy: didYouUnderstandOtherPlayerStrategy,
            otherPlayerRating: otherPlayerRating,
            connectionFeeling: connectionFeeling
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Questionnaire answers saved successfully!', data);
            // Reset the form after successful submission
            document.getElementById('questionnaireContainer').style.display = 'none';
            document.querySelectorAll('input[name="quantitativeStrategyUsed"]').forEach(input => input.checked = false);
            document.querySelectorAll('input[name="quantitativeOtherPlayerStrategy"]').forEach(input => input.checked = false);
            document.querySelectorAll('input[name="qualitativeStrategyUsed"]').forEach(input => input.checked = false);
            document.querySelectorAll('input[name="qualitativeOtherPlayerStrategy"]').forEach(input => input.checked = false);
            document.getElementById('otherPlayerUnderstoodYourStrategies').value = '';
            document.getElementById('didYouUnderstandOtherPlayerStrategy').value = '';
            document.getElementById('otherPlayerRating').value = '';
            document.getElementById('connectionFeeling').value = '';
            document.getElementById('errorMessageQuestionnaire').style.display = 'none';
        } else {
            console.error('Failed to save questionnaire answers.', data);
        }
    })
    .catch(err => {
        console.error('Error saving questionnaire answers:', err);
    });

    // Optionally show a thank-you container after submission
    document.getElementById('questionnaireContainer').style.display = 'none';
    document.getElementById('thankYouContainer').style.display = 'block';

    fetchGameStats();
});

document.getElementById('restartButtonTwo').addEventListener('click', async function (event) {
    document.getElementById('thankYouContainer').style.display = 'none';
    console.log("Restarting the game...");
    console.log('gameConfigOrder:', localStorage.getItem('gameConfigOrder'));
    resetTheGame();
    console.log("Restarting the game...");
    console.log('gameConfigOrder:', localStorage.getItem('gameConfigOrder'));
    initialiseGameSetup();
});
// END QUESTIONNAIRE LOGIC