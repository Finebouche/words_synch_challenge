
// CONFETTI
function create_confetti(i) {
    const wrapper = document.getElementById('confetti-wrapper');

    const width = Math.random() * 8;
    const height = width * 0.4;
    const colourIdx = Math.ceil(Math.random() * 3);
    let colour;
    switch (colourIdx) {
        case 1:
        colour = 'yellow';
        break;
        case 2:
        colour = 'blue';
        break;
        default:
        colour = 'red';
    }
    const confetti = document.createElement('div');
    confetti.className = `confetti ${colour}`;
    confetti.style.width = `${width}px`;
    confetti.style.height = `${height}px`;
    confetti.style.top = `${-Math.random() * 20}%`;
    confetti.style.left = `${Math.random() * 100}%`;
    confetti.style.opacity = Math.random() + 0.5;
    confetti.style.transform = `rotate(${Math.random() * 360}deg)`;
    wrapper.appendChild(confetti);

    drop(confetti);
}

function drop(element) {
    const endTop = 99;
    const endLeft = Math.max(Math.min(parseInt(element.style.left) + (Math.random() - 0.5)* 30, 99), 0);
    element.animate([
        { top: element.style.top, left: element.style.left },
        { top: `${endTop}%`, left: `${endLeft}%` }
    ], {
        duration: Math.random() * 2000 + 2000,
        fill: 'forwards'
    }).onfinish = function() {
        reset(element);
    };
}

function reset(element) {
    element.style.top = `${-Math.random() * 20}%`;
    element.style.left = `${Math.random() * 100}%`;
    drop(element);
}

function winGame() {
    document.getElementById('gameRestart').style.display = 'flex';
    document.getElementById('winMessage').style.display = 'block';
    document.getElementById('gameInput').style.display = 'none';

    for (let i = 0; i < 150; i++) {
        create_confetti(i);
    }
}
// END CONFETTI

// RAIN ANIMATION
function loseGame() {
    document.getElementById('gameRestart').style.display = 'flex';
    document.getElementById('lossMessage').style.display = 'block';
    document.getElementById('gameInput').style.display = 'none';

    // Clear out everything
    document.querySelectorAll('.rain-wrapper').forEach(function(rainElement) {
        rainElement.innerHTML = '';
    });

    var increment = 0;
    var drops = "";
    var backDrops = "";

    while (increment < 95) {
        // Couple of random numbers to use for various randomizations
        // Random number between 98 and 1
        var randoHundo = Math.floor(Math.random() * 98);
        // Random number between 5 and 2
        var randoFiver = Math.floor(Math.random() * 3 + 2);
        // Increment
        increment += randoFiver;
        // Add in a new raindrop with various randomizations to certain CSS properties
        drops = '<div class="drop" style="left: ' + increment + '%; bottom: ' + (randoFiver + randoFiver - 1 + 100) + '%; animation-delay: 0.' + randoHundo + 's; animation-duration: 0.5' + randoHundo + 's;"><div class="stem" style="animation-delay: 0.' + randoHundo + 's; animation-duration: 0.5' + randoHundo + 's;"></div><div class="splat" style="animation-delay: 0.' + randoHundo + 's; animation-duration: 0.5' + randoHundo + 's;"></div></div>';
        backDrops = '<div class="drop" style="right: ' + increment + '%; bottom: ' + (randoFiver + randoFiver - 1 + 100) + '%; animation-delay: 0.' + randoHundo + 's; animation-duration: 0.5' + randoHundo + 's;"><div class="stem" style="animation-delay: 0.' + randoHundo + 's; animation-duration: 0.5' + randoHundo + 's;"></div><div class="splat" style="animation-delay: 0.' + randoHundo + 's; animation-duration: 0.5' + randoHundo + 's;"></div></div>';
        document.querySelector('.rain-wrapper.front-row').innerHTML += drops;
        document.querySelector('.rain-wrapper.back-row').innerHTML += backDrops;
    }
}
// END RAIN ANIMATION