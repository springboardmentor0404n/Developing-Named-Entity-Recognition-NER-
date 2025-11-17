const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const chatOutput = document.getElementById("chatOutput");
const sendBtn = document.getElementById("sendBtn");
const queryInput = document.getElementById("queryInput");

let uploadedText = "";

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return alert("Please select a file!");

    const formData = new FormData();
    formData.append("file", file);

    chatOutput.innerHTML += `<div><b>Uploading:</b> ${file.name}</div>`;

    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    uploadedText = data.text;

    chatOutput.innerHTML += `<div><b>System:</b> File uploaded successfully and text extracted.</div>`;
});

sendBtn.addEventListener("click", async () => {
    const query = queryInput.value.trim();
    if (!query) return;

    chatOutput.innerHTML += `<div><b>You:</b> ${query}</div>`;
    queryInput.value = "";

    const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, text: uploadedText })
    });

    const data = await res.json();
    chatOutput.innerHTML += `<div><b>Gemini:</b> ${data.response}</div>`;
});
