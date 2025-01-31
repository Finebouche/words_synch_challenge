// LANGUAGE OPTIONS
document.getElementById('language-selector').addEventListener('click', function () {
    var languageOptions = document.getElementById('languageOptions');
    languageOptions.style.display = languageOptions.style.display === 'block' ? 'none' : 'block';
});

function getTranslation(key) {
    if (translations && key in translations) {
        return translations[key];
    }
    return key; // Return the key itself if translation is not found
}
document.addEventListener('DOMContentLoaded', function () {
    loadLanguage('en'); // Default to English or use browser's language setting
});
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

document.querySelectorAll('#languageOption').forEach(function (element) {
    element.addEventListener('click', function () {
        const selectedLang = this.getAttribute('data-lang');
        loadLanguage(selectedLang);

        // Update current language display
        document.getElementById('currentLanguage').innerHTML = this.innerHTML;
    });
});


document.addEventListener('click', function (event) {
    var currentUser = document.getElementById('user-profile-selector');
    var currentLanguage = document.getElementById('language-selector');

    // Check if the click is outside the languageOptions and currentLanguage
    if (!currentLanguage.contains(event.target)) {
        languageOptions.style.display = 'none';
    }
    // Check if the click is outside the userOptions
    if (!currentUser.contains(event.target)) {
        userOptions.style.display = 'none';
    }
});
// END LANGUAGE OPTIONS


