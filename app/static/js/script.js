document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const textInput = document.getElementById('text-input');
    const outputDiv = document.getElementById('output');
    const loader = document.querySelector('.loader');

    generateBtn.addEventListener('click', async () => {
        // Show loading state
        generateBtn.disabled = true;
        loader.classList.add('visible');
        generateBtn.classList.add('pulse');

        // Get the input value
        const maxTokens = textInput.value || 1000;

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    max_new_tokens: parseInt(maxTokens)
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            outputDiv.textContent = data.result;
        } catch (error) {
            outputDiv.textContent = 'Error: ' + error.message;
            console.error('Error:', error);
        } finally {
            // Hide loading state
            generateBtn.disabled = false;
            loader.classList.remove('visible');

            // Remove pulse class after animation completes
            setTimeout(() => {
                generateBtn.classList.remove('pulse');
            }, 500);
        }
    });
});