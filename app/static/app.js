
// ============================================
// MTS ENERGYTOOL - CHAT APPLICATION
// ============================================


// ============================================
// DOM ELEMENTS
// ============================================

const chatArea = document.getElementById("chat-area");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const welcomeScreen = document.getElementById("welcome-screen");
const sessionList = document.getElementById("session-list");

// Lead form elements
const leadToggleBtn = document.getElementById("lead-toggle-btn");
const leadForm = document.getElementById("lead-form");
const leadSubmitBtn = document.getElementById("lead-submit-btn");
const leadStatus = document.getElementById("lead-status");


// ============================================
// APPLICATION STATE
// ============================================

let conversationHistory = [];
let currentSessionId = null;
let isSending = false;


// ============================================
// DISPLAY A MESSAGE
// ============================================

function displayMessage(role, content, sources = [], responseTime = null) {

    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageContent.textContent = content;

    messageDiv.appendChild(messageContent);


    // DISPLAY SOURCES

    if (role === "assistant" && Array.isArray(sources) && sources.length > 0) {

        const sourcesDiv = document.createElement("div");
        sourcesDiv.className = "message-sources";

        const sourcesTitle = document.createElement("strong");
        sourcesTitle.textContent = "Источники:";
        sourcesDiv.appendChild(sourcesTitle);

        sources.forEach((source) => {

            const sourceItem = document.createElement("div");
            sourceItem.className = "source-item";

            let sourceText = "";

            if (typeof source === "string") {
                sourceText = source;
            } else if (source && typeof source === "object") {
                sourceText =
                    source.source ||
                    source.path ||
                    source.title ||
                    source.name ||
                    JSON.stringify(source);
            }

            sourceItem.textContent = sourceText;
            sourcesDiv.appendChild(sourceItem);
        });

        messageDiv.appendChild(sourcesDiv);
    }


    // DISPLAY RESPONSE TIME

    if (role === "assistant" && responseTime !== null) {

        const timeDiv = document.createElement("div");
        timeDiv.className = "response-time";

        timeDiv.textContent = `Время ответа: ${responseTime} мс`;

        messageDiv.appendChild(timeDiv);
    }

    chatArea.appendChild(messageDiv);

    chatArea.scrollTop = chatArea.scrollHeight;
}


// ============================================
// LOADING INDICATOR
// ============================================

function showLoading() {

    const loadingDiv = document.createElement("div");

    loadingDiv.id = "loading-indicator";
    loadingDiv.className = "message assistant loading";

    loadingDiv.textContent = "МТС EnergyTool готовит ответ...";

    chatArea.appendChild(loadingDiv);

    chatArea.scrollTop = chatArea.scrollHeight;
}


function removeLoading() {

    const loadingDiv = document.getElementById("loading-indicator");

    if (loadingDiv) {
        loadingDiv.remove();
    }
}


// ============================================
// SHOW WELCOME SCREEN
// ============================================

function showWelcomeScreen() {

    chatArea.innerHTML = "";

    if (welcomeScreen) {
        chatArea.appendChild(welcomeScreen);
        welcomeScreen.style.display = "flex";
    }
}


// ============================================
// SEND QUESTION TO FASTAPI
// ============================================

async function sendQuestion(question) {

    if (!question || !question.trim() || isSending) {
        return;
    }

    isSending = true;

    if (welcomeScreen) {
        welcomeScreen.style.display = "none";
    }

    // Display user's question
    displayMessage("user", question);

    // Save previous conversation history
    const previousHistory = [...conversationHistory];

    // Add current question to history
    conversationHistory.push({
        role: "user",
        content: question
    });

    messageInput.value = "";
    messageInput.style.height = "auto";

    sendBtn.disabled = true;

    showLoading();

    try {

        const response = await fetch("/api/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                question: question,

                history: previousHistory,

                session_id: currentSessionId

            })

        });


        if (!response.ok) {

            const errorData = await response.json().catch(() => ({}));

            throw new Error(
                errorData.detail || "Не удалось получить ответ от сервера."
            );
        }


        const data = await response.json();

        removeLoading();


        // Store the session ID returned by the backend

        if (data.session_id !== undefined && data.session_id !== null) {
            currentSessionId = data.session_id;
        }


        // Store assistant response

        conversationHistory.push({
            role: "assistant",
            content: data.answer
        });


        // Display assistant response

        displayMessage(
            "assistant",
            data.answer,
            data.sources || [],
            data.response_time_ms
        );


        // Refresh conversation sidebar

        await loadSessions();

    } catch (error) {

        removeLoading();

        console.error("Chat error:", error);

        displayMessage(
            "assistant",
            "Произошла ошибка при обработке запроса. Пожалуйста, попробуйте ещё раз."
        );

        // Remove failed question from local history

        conversationHistory = previousHistory;

    } finally {

        isSending = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}


// ============================================
// LOAD ALL SAVED CONVERSATIONS
// ============================================

async function loadSessions() {

    if (!sessionList) {
        return;
    }

    try {

        const response = await fetch("/api/sessions");

        if (!response.ok) {
            throw new Error("Не удалось загрузить историю диалогов.");
        }

        const sessions = await response.json();

        sessionList.innerHTML = "";


        if (!sessions.length) {

            const emptyMessage = document.createElement("p");

            emptyMessage.className = "history-empty";
            emptyMessage.textContent = "Пока нет диалогов";

            sessionList.appendChild(emptyMessage);

            return;
        }


        sessions.forEach((session) => {

            const sessionButton = document.createElement("button");

            sessionButton.type = "button";
            sessionButton.className = "session-item";

            if (session.id === currentSessionId) {
                sessionButton.classList.add("active");
            }

            sessionButton.textContent = session.title || "Новый диалог";

            sessionButton.title = session.title || "Новый диалог";

            sessionButton.addEventListener("click", () => {
                openSession(session.id);
            });

            sessionList.appendChild(sessionButton);
        });

    } catch (error) {

        console.error("Session loading error:", error);
    }
}


// ============================================
// OPEN AN EXISTING CONVERSATION
// ============================================

async function openSession(sessionId) {

    if (isSending) {
        return;
    }

    try {

        const response = await fetch(`/api/sessions/${sessionId}`);

        if (!response.ok) {
            throw new Error("Не удалось открыть диалог.");
        }

        const session = await response.json();


        // Set active session

        currentSessionId = session.id;


        // Clear current messages

        chatArea.innerHTML = "";

        conversationHistory = [];


        // Restore saved messages

        if (session.messages && session.messages.length > 0) {

            session.messages.forEach((message) => {

                displayMessage(
                    message.role,
                    message.content
                );

                conversationHistory.push({
                    role: message.role,
                    content: message.content
                });

            });

        } else {

            showWelcomeScreen();
        }


        // Highlight active conversation

        await loadSessions();

        messageInput.focus();

    } catch (error) {

        console.error("Open session error:", error);

        alert("Не удалось открыть этот диалог. Попробуйте ещё раз.");
    }
}


// ============================================
// CREATE A NEW CONVERSATION
// ============================================

function startNewChat() {

    if (isSending) {
        return;
    }

    currentSessionId = null;

    conversationHistory = [];

    showWelcomeScreen();

    loadSessions();

    messageInput.value = "";

    messageInput.style.height = "auto";

    messageInput.focus();

    // Clear lead form status and fields

    if (leadForm) {
        leadForm.reset();
        leadForm.hidden = true;
    }

    if (leadStatus) {
        leadStatus.textContent = "";
    }
}


// ============================================
// CHAT FORM SUBMISSION
// ============================================

chatForm.addEventListener("submit", async (event) => {

    event.preventDefault();

    const question = messageInput.value.trim();

    await sendQuestion(question);
});


// ============================================
// NEW CHAT BUTTON
// ============================================

newChatBtn.addEventListener("click", () => {

    startNewChat();
});


// ============================================
// SUGGESTED QUESTIONS
// ============================================

document.querySelectorAll(".suggestion").forEach((button) => {

    button.addEventListener("click", async () => {

        // Ignore the lead form toggle button
        if (button.id === "lead-toggle-btn") {
            return;
        }

        const question = button.textContent.trim();

        await sendQuestion(question);
    });
});


// ============================================
// TEXTAREA AUTO RESIZE
// ============================================

messageInput.addEventListener("input", () => {

    messageInput.style.height = "auto";

    messageInput.style.height =
        Math.min(messageInput.scrollHeight, 150) + "px";
});


// ============================================
// ENTER TO SEND
// SHIFT + ENTER FOR NEW LINE
// ============================================

messageInput.addEventListener("keydown", (event) => {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        chatForm.requestSubmit();
    }
});


// ============================================
// CUSTOMER LEAD COLLECTION
// ============================================

// Open and close the lead form

if (leadToggleBtn && leadForm) {

    leadToggleBtn.addEventListener("click", () => {

        leadForm.hidden = !leadForm.hidden;

        if (leadStatus) {
            leadStatus.textContent = "";
        }

        if (!currentSessionId && !leadForm.hidden) {

            leadStatus.textContent =
                "Сначала задайте вопрос консультанту, чтобы создать диалог. Затем вы сможете отправить заявку.";

            leadStatus.style.color = "#b45309";
        }
    });
}


// Submit lead information to FastAPI

if (leadForm) {

    leadForm.addEventListener("submit", async (event) => {

        event.preventDefault();

        if (!currentSessionId) {

            leadStatus.textContent =
                "Сначала задайте вопрос консультанту, чтобы создать диалог.";

            leadStatus.style.color = "#b91c1c";

            return;
        }

        const formData = new FormData(leadForm);

        const leadPayload = {

            session_id: currentSessionId,

            full_name: String(
                formData.get("full_name") || ""
            ).trim(),

            phone: String(
                formData.get("phone") || ""
            ).trim(),

            company: String(
                formData.get("company") || ""
            ).trim(),

            summary: String(
                formData.get("summary") || ""
            ).trim()
        };


        // Validate required fields

        if (
            !leadPayload.full_name ||
            !leadPayload.phone ||
            !leadPayload.company
        ) {

            leadStatus.textContent =
                "Пожалуйста, заполните ФИО, телефон и название компании.";

            leadStatus.style.color = "#b91c1c";

            return;
        }


        leadSubmitBtn.disabled = true;

        leadStatus.textContent = "Отправка заявки...";

        leadStatus.style.color = "";


        try {

            const response = await fetch("/api/leads", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(leadPayload)

            });


            const data = await response.json().catch(() => ({}));


            if (!response.ok) {

                throw new Error(
                    data.detail || "Не удалось отправить заявку."
                );
            }


            // Successful submission

            leadStatus.textContent =
                data.message || "Заявка успешно сохранена.";

            leadStatus.style.color = "#15803d";

            leadForm.reset();

            console.log("Lead saved successfully:", data);


        } catch (error) {

            console.error("Lead submission error:", error);

            leadStatus.textContent =
                error.message || "Ошибка отправки заявки. Попробуйте ещё раз.";

            leadStatus.style.color = "#b91c1c";

        } finally {

            leadSubmitBtn.disabled = false;
        }
    });
}


// ============================================
// BACKEND CONNECTION CHECK
// ============================================

async function checkBackendConnection() {

    try {

        const response = await fetch("/api/health");

        if (!response.ok) {
            throw new Error("Backend is unavailable.");
        }

        const data = await response.json();

        console.log("MTS EnergyTool backend connected:", data);

    } catch (error) {

        console.error("Backend connection error:", error);
    }
}


// ============================================
// APPLICATION INITIALIZATION
// ============================================

async function initializeApplication() {

    await checkBackendConnection();

    await loadSessions();

    console.log("MTS EnergyTool chat application initialized.");
}


// ============================================
// START APPLICATION
// ============================================

initializeApplication();
