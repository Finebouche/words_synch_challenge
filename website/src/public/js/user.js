/*******************************************************
 * Utility Functions
 *******************************************************/
function generateNewPlayerID() {
  const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  return Array.from({ length: 8 }, () =>
    characters.charAt(Math.floor(Math.random() * characters.length))
  ).join('');
}

/** Retrieves a value from localStorage and handles potential 'null' string values.
 * If the item is 'null' or not set, returns an empty string instead.
 * @param {string} key - The key to retrieve from sessionStorage.
 * @return {string} - The value from localStorage or an empty string if not found or 'null'.
 */
function getLocalStorageValue(key) {
  const value = sessionStorage.getItem(key);
  return (value === 'null' || value === null) ? '' : value;
}


/** Sets the value of a form input or text content of an element based on its ID.
 * @param {string} key - The localStorage key associated with the element.
 * @param {string} elementId - The ID of the DOM element to update.
 * @param {boolean} isText - If true, sets textContent; otherwise, sets value.
 */
function populateElementFromStorage(key, elementId, isText = false) {
  const value = getLocalStorageValue(key);
  const element = document.getElementById(elementId);
  if (element) {
    if (isText) {
      element.textContent = value;
    } else {
      element.value = value;
    }
  }
}

function fetchGameStats() {
    let playerId = sessionStorage.getItem('connectedPlayerId') || getLocalStorageValue('newPlayerId');
    fetch(`/game/number_games`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ playerId: playerId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('gamesPlayed').style.display = 'block';
            document.getElementById('gamesPlayed').innerHTML = `
                <strong class="games-played-title">Games Played</strong>
                <div class="count-cell">
                    😊  human  : ${data.gamesPlayedAgainstHuman}
                </div>
            
                <div class="count-cell">
                    🤖  ai  : ${data.gamesPlayedAgainstBot}
                </div>
            `;
            // update local storage with the new game stats
            sessionStorage.setItem('gamesPlayedAgainstHuman', data.gamesPlayedAgainstHuman);
            sessionStorage.setItem('gamesPlayedAgainstBot', data.gamesPlayedAgainstBot);
            // if the total number of game played is bigger than 10, show the return to prolific button
            if (parseInt(sessionStorage.getItem("gamesPlayedAgainstHuman")) >= 5 && sessionStorage.getItem("gamesPlayedAgainstBot") >= 5) {
                document.getElementById('returnToProlific').style.display = 'block';
            }
        }
    })
    .catch(error => console.error('Failed to fetch game stats:', error));
}

/*******************************************************
    * Event Listeners
 * ******************************************************/

window.addEventListener('DOMContentLoaded', function() {
    let playerId = getLocalStorageValue('playerId');
    sessionStorage.setItem('newPlayerId', generateNewPlayerID());

    if (playerId) {
        fetchGameStats();

        // User is recognized in localStorage, prepare and show user profile interface.
        document.getElementById('login').style.display = 'none';
        document.getElementById('signin').style.display = 'none';
        document.getElementById('parameters').style.display = 'flex';

        // Populate form fields and user identifiers from localStorage
        populateElementFromStorage('pseudonym', 'pseudonymInput');
        populateElementFromStorage('playerId', 'userId', true);
        populateElementFromStorage('ageGroup', 'ageGroupInput');
        populateElementFromStorage('gender', 'genderInput');
        populateElementFromStorage('region', 'regionInput');
        populateElementFromStorage('llmKnowledge', 'llmKnowledgeInput');

        // Set user's display name in the user interface
        const displayName = getLocalStorageValue('pseudonym') || playerId;
        if (displayName) {
            document.getElementById('currentUser').innerHTML = '<span role="img" aria-label="User">&#x1F464;</span> ' + displayName;
        }
    } else {
        // No playerId in storage, show login interface
        document.getElementById('login').style.display = 'flex';
        document.getElementById('parameters').style.display = 'none';
    }
    console.log('Player ID:', playerId);
});

document.getElementById('user-profile-selector').addEventListener('click', function () {
    let userOptions = document.getElementById('userOptions');
    document.getElementById('red-arrow-tooltip').style.display = 'none';
    userOptions.style.display = "block";
});

document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const userId = document.getElementById('current-password').value;
    fetch('/auth/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ playerId: userId })
    })
    .then(response => {
        if (response.status === 404) {
            console.log("User not found.");
            document.getElementById('errorBanner').style.display = 'flex'; // Show the banner
            document.getElementById('llmSelect').value = '';
            document.getElementById('submitWord').disabled = true;
            document.getElementById('startGame').style.display = 'none';
            return null; // Stop further processing
        } else {
            console.log("User found.");
            document.getElementById('parameters').style.display = 'flex';
            document.getElementById('login').style.display = 'none';
            document.getElementById('signin').style.display = 'none';
            return response.json()
        }
    })
    .then(data => {
        // Store the data in variables (or directly set them in the UI)
        const pseudonym = data.pseudonym;
        const playerId  = data.playerId;
        const ageGroup  = data.ageGroup;
        const gender    = data.gender;
        const region    = data.region;
        const llmKnowledge = data.llmKnowledge;

        // Store data in localStorage
        sessionStorage.setItem('pseudonym', pseudonym);
        sessionStorage.setItem('connectedPlayerId', playerId);
        sessionStorage.setItem('ageGroup', ageGroup);
        sessionStorage.setItem('gender', gender);
        sessionStorage.setItem('region', region);
        sessionStorage.setItem('llmKnowledge', llmKnowledge);

        // Populate the DOM elements
        document.getElementById('pseudonymInput').value = pseudonym;
        document.getElementById('userId').textContent = playerId;
        document.getElementById('ageGroupInput').value = ageGroup || ''; // fallback to empty if not provided
        document.getElementById('genderInput').value = gender || '';
        document.getElementById('regionInput').value = region || '';
        document.getElementById('llmKnowledgeInput').value = llmKnowledge || '';
        const displayName = getLocalStorageValue('pseudonym') || playerId;
        if (displayName) {
          document.getElementById('currentUser').innerHTML =
            '<span role="img" aria-label="User">&#x1F464;</span> ' + displayName;
        }
    }).then(() => {
        fetchGameStats();
    });

});

document.getElementById('createPlayer').addEventListener('click', function() {
    document.getElementById('parameters').style.display = 'none';
    document.getElementById('login').style.display = 'none';
    document.getElementById('signin').style.display = 'flex';
    document.getElementById('newUserId').textContent = getLocalStorageValue('newPlayerId');
});

document.getElementById('copyId').addEventListener('click', function() {
    let playerId = getLocalStorageValue('newPlayerId');
    navigator.clipboard.writeText(playerId).then(function() {
        fetch('/auth/create', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ playerId: playerId })
        })

        .then(response => response.json())
        .then(data => {
            document.getElementById('copyId').style.display = 'none';
            document.getElementById('copiedId').style.display = 'block';
            document.getElementById('goLogin').style.display = 'block';
            document.getElementById('goLogin').disabled = false;
        });
    });
});


document.getElementById('goLogin').addEventListener('click', function() {
    let playerId = getLocalStorageValue('playerId');
    document.getElementById('parameters').style.display = 'none';
    document.getElementById('login').style.display = 'flex';
    document.getElementById('signin').style.display = 'none';
    document.getElementById('userId').textContent = playerId;
    sessionStorage.setItem('newPlayerId', generateNewPlayerID());
});


document.getElementById('updateProfile').addEventListener('click', function() {
    const pseudonym   = document.getElementById('pseudonymInput').value;
    const ageGroup    = document.getElementById('ageGroupInput').value;
    const gender      = document.getElementById('genderInput').value;
    const region      = document.getElementById('regionInput').value;
    const llmKnowledge = document.getElementById('llmKnowledgeInput').value;
    const playerId = getLocalStorageValue('playerId');

    fetch('/auth/update-profile', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            playerId: playerId,
            pseudonym: pseudonym || '',
            ageGroup: ageGroup || '',
            gender: gender || '',
            region: region || '',
            llmKnowledge: llmKnowledge || ''
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Profile update failed');
        }
        return response.text(); // or response.json() if the server returns JSON
    })
    .then(message => {
        console.log('Profile update message:', message);
        sessionStorage.setItem('pseudonym', pseudonym);
        sessionStorage.setItem('ageGroup', ageGroup);
        sessionStorage.setItem('gender', gender);
        sessionStorage.setItem('region', region);
        sessionStorage.setItem('llmKnowledge', llmKnowledge);

        const displayName = getLocalStorageValue('pseudonym') || playerId;
        if (displayName) {
          document.getElementById('currentUser').innerHTML =
            '<span role="img" aria-label="User">&#x1F464;</span> ' + displayName;
        }

    })
    .then(() => {
        // add green-button class to the updateProfile button
        document.getElementById('updateProfile').classList.add('green-button');
        // Wait 1 second and remove the green-button class
        setTimeout(() => {
            document.getElementById('updateProfile').classList.remove('green-button');
        }, 3000);

    })
    .catch(error => {
        console.error('Error:', error);
        // Optionally show an error message to the user
    });
});

document.getElementById('logoutPlayer').addEventListener('click', function() {
    // Remove user data from localStorage
    sessionStorage.clear();

    sessionStorage.setItem('newPlayerId', generateNewPlayerID());
    sessionStorage.removeItem('connectedPlayerId');

    // Reset UI
    document.getElementById('parameters').style.display = 'none';
    document.getElementById('gamesPlayed').style.display = 'none';
    document.getElementById('login').style.display = 'flex';
    document.getElementById('signin').style.display = 'none';
    const currentUserDiv = document.getElementById('currentUser');
    currentUserDiv.innerHTML = '<span role="img" aria-label="User">&#x1F464;</span> ' + 'Log In';
});
