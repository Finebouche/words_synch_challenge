
// START QUESTIONNAIRE LOGIC
document.getElementById('questionsButton').addEventListener('click', function() {
    document.getElementById('questionnaireContainer').style.display = 'block';
    cleanPreviousGameArea()
});

function limitCheckboxSelection(groupName, maxSelection) {
    const checkboxes = document.querySelectorAll(`input[name="${groupName}"]`);
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function () {
            const checkedBoxes = Array.from(checkboxes).filter(cb => cb.checked);

            if (checkedBoxes.length > maxSelection) {
                // Uncheck the first selected checkbox
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


document.getElementById('submitQuestionnaire').addEventListener('click', function() {
    const quantitativeStrategyUsed = document.querySelector('input[name="quantitativeStrategyUsed"]:checked');

    const qualitativeStrategyUsed = Array.from(document.querySelectorAll('input[name="qualitativeStrategyUsed"]:checked'))
                              .map(cb => cb.value);
    const quantitativeOtherPlayerStrategy = Array.from(document.querySelectorAll('input[name="quantitativeOtherPlayerStrategy"]:checked'))
                                     .map(cb => cb.value);
    const qualitativeOtherPlayerStrategy = document.querySelector('input[name="qualitativeOtherPlayerStrategy"]:checked');

    const otherPlayerUnderstood = document.getElementById('otherPlayerRating').value;
    const didYouUnderstandOtherPlayerStrategy = document.getElementById('otherPlayerRating').value;
    const otherPlayerRating = document.getElementById('otherPlayerRating').value;
    const connectionFeeling = document.getElementById('otherPlayerRating').value;


    // Post the data to the server
    fetch('/game/answers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            gameId: gameId,
            role: myRole,
            playerId: getLocalStorageValue('playerId'),
            quantitativeStrategyUsed: quantitativeStrategyUsed,
            qualitativeStrategyUsed: qualitativeStrategyUsed,
            quantitativeOtherPlayerStrategy: quantitativeOtherPlayerStrategy,
            qualitativeOtherPlayerStrategy: qualitativeOtherPlayerStrategy,
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
            document.getElementById('questionnaireContainer').style.display = 'none';
            document.querySelectorAll('input[name="quantitativeStrategyUsed"]').forEach(input => input.checked = false);
            document.querySelectorAll('input[name="qualitativeStrategyUsed"]').forEach(input => input.checked = false);
            document.querySelectorAll('input[name="quantitativeOtherPlayerStrategy"]').forEach(input => input.checked = false);
            document.querySelectorAll('input[name="qualitativeOtherPlayerStrategy"]').forEach(input => input.checked = false);
            document.getElementById('otherPlayerUnderstoodYourStrategies').value = '1';
            document.getElementById('didYouUnderstandOtherPlayerStrategy').value = '1';
            document.getElementById('otherPlayerRating').value = '1';
            document.getElementById('connectionFeeling').value = '1';
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

    resetTheGame();

});
// END QUESTIONNAIRE LOGIC
