const form = document.getElementById("predictForm");
const loader = document.getElementById("loader");
const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");

form.addEventListener("submit", async function (e) {
    e.preventDefault();

    loader.classList.remove("hidden");
    emptyState.classList.add("hidden");
    resultContent.classList.add("hidden");

    const button = form.querySelector("button");
    button.disabled = true;
    button.innerHTML = `<i data-lucide="loader-circle"></i> Analyzing Soil...`;
    lucide.createIcons();

    const formData = new FormData(form);

    const payload = {
        nitrogen: formData.get("nitrogen"),
        phosphorus: formData.get("phosphorus"),
        potassium: formData.get("potassium"),
        temperature: formData.get("temperature"),
        humidity: formData.get("humidity"),
        ph: formData.get("ph"),
        rainfall: formData.get("rainfall")
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        setTimeout(() => {
            loader.classList.add("hidden");
            resultContent.classList.remove("hidden");

            document.getElementById("cropImage").src = data.image;
            document.getElementById("cropName").textContent = data.crop;
            document.getElementById("cropReason").textContent = data.reason;
            document.getElementById("cropFertilizer").textContent = data.fertilizer;
            document.getElementById("cropTips").textContent = data.tips;

            button.disabled = false;
            button.innerHTML = `<i data-lucide="sparkles"></i> Analyze Again`;
            lucide.createIcons();

            resultContent.classList.add("pop-in");

        }, 900);

    } catch (error) {
        loader.classList.add("hidden");
        emptyState.classList.remove("hidden");

        button.disabled = false;
        button.innerHTML = `<i data-lucide="sparkles"></i> Analyze Best Crop`;
        lucide.createIcons();

        alert("Something went wrong while analyzing crop recommendation.");
    }
});