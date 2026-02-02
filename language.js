document.getElementById("language-icon").addEventListener("click", function () {
    let langSelect = document.getElementById("language-select");
    langSelect.classList.toggle("hidden");
});

document.getElementById("language-select").addEventListener("change", function () {
    let selectedLang = this.value;
    updateLanguage(selectedLang);
    this.classList.add("hidden"); // Hide dropdown after selection
});

function updateLanguage(lang) {
    document.querySelectorAll(".translate").forEach(el => {
        el.textContent = el.getAttribute(`data-${lang}`);
    });
}


