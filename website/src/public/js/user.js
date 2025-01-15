
// USER OPTIONS
document.getElementById('user-profile-selector').addEventListener('click', function () {
    var userOptions = document.getElementById('userOptions');
    userOptions.style.display = 'block';
});

function generatePlayerID() {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    return Array.from({ length: 8 }, () => characters.charAt(Math.floor(Math.random() * characters.length))).join('');
}
var playerId = generatePlayerID();
var pseudonym = '';
console.log('Player ID:', playerId);

document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent full page reload

    const userId = document.getElementById('userIdInput').value;
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
        pseudonym = data.pseudonym;
        playerId = data.playerId;
        document.getElementById('pseudonymInput').textContent = pseudonym;
        document.getElementById('userId').textContent = playerId;
        const currentUserDiv = document.getElementById('currentUser');
        currentUserDiv.innerHTML = '<span role="img" aria-label="User">&#x1F464;</span> ' + (pseudonym || userId);
    });
});

document.getElementById('createPlayer').addEventListener('click', function() {
    document.getElementById('parameters').style.display = 'none';
    document.getElementById('login').style.display = 'none';
    document.getElementById('signin').style.display = 'flex';
    document.getElementById('newUserId').textContent = playerId;
});

document.getElementById('copyId').addEventListener('click', function() {
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
    document.getElementById('parameters').style.display = 'none';
    document.getElementById('login').style.display = 'flex';
    document.getElementById('signin').style.display = 'none';
    document.getElementById('userId').textContent = playerId;
});


document.getElementById('updateProfile').addEventListener('click', function() {
    const pseudonym = document.getElementById('pseudonymInput').value;
    const ageGroup = document.getElementById('ageGroupInput').value;
    const gender = document.getElementById('genderInput').value;
    const region = document.getElementById('regionInput').value;
    const llmKnowledge = document.getElementById('llmKnowledgeInput').value;

    // Make a request to the new /auth/update-profile route
    fetch('/auth/update-profile', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            playerId: playerId,        // from your existing code
            pseudonym: pseudonym || '',  // or undefined if empty
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
        return response.text();
    })
    .then(message => {
        console.log(message);
        const currentUserDiv = document.getElementById('currentUser');
        currentUserDiv.innerHTML = '<span role="img" aria-label="User">&#x1F464;</span> ' + pseudonym;
    })
    .catch(error => {
        console.error('Error:', error);
        // Show error message to the user
    });
});

document.getElementById('logoutPlayer').addEventListener('click', function() {
    document.getElementById('parameters').style.display = 'none';
    document.getElementById('login').style.display = 'flex';
    document.getElementById('signin').style.display = 'none';
    const currentUserDiv = document.getElementById('currentUser');
    currentUserDiv.innerHTML = '<span role="img" aria-label="User">&#x1F464;</span> ' + 'Log In';
});
// END USER OPTIONS
