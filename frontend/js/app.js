"use strict";

/*
 * ABC Tech Smart Bot
 * Step 14 - Frontend application logic
 *
 * Flow:
 *
 * User question
 *      ↓
 * Validate input
 *      ↓
 * POST /api/chat
 *      ↓
 * Flask RAG API
 *      ↓
 * Answer + sources
 *      ↓
 * Render accessible UI
 */

const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const sendButtonText = document.getElementById("send-button-text");

const messageArea = document.getElementById("message-area");
const statusMessage = document.getElementById("status-message");
const errorMessage = document.getElementById("error-message");

const API_URL = "/api/chat";


/* ---------------------------------------------------------
   Utility functions
   --------------------------------------------------------- */

function clearError() {
    errorMessage.hidden = true;
    errorMessage.textContent = "";
}


function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
}


function setLoading(isLoading) {
    sendButton.disabled = isLoading;
    questionInput.disabled = isLoading;

    if (isLoading) {
        document.body.classList.add("is-loading");

        sendButtonText.innerHTML =
            '<span class="loading-indicator">Thinking...</span>';

        statusMessage.textContent =
            "The assistant is processing your question.";
    } else {
        document.body.classList.remove("is-loading");

        sendButtonText.textContent = "Ask Smart Bot";

        statusMessage.textContent = "";
    }
}


function scrollMessagesToBottom() {
    messageArea.scrollTop = messageArea.scrollHeight;
}


/* ---------------------------------------------------------
   Safe text rendering
   --------------------------------------------------------- */

function createTextElement(tagName, text, className = "") {
    const element = document.createElement(tagName);

    if (className) {
        element.className = className;
    }

    element.textContent = text;

    return element;
}


/* ---------------------------------------------------------
   Message rendering
   --------------------------------------------------------- */

function addUserMessage(question) {
    const article = document.createElement("article");
    article.className = "message message-user";

    const content = document.createElement("div");
    content.className = "message-content";

    const label = createTextElement(
        "p",
        "You",
        "message-label"
    );

    const text = createTextElement(
        "p",
        question
    );

    content.appendChild(label);
    content.appendChild(text);

    article.appendChild(content);
    messageArea.appendChild(article);

    scrollMessagesToBottom();
}


function addBotMessage(answer, sources = []) {
    const article = document.createElement("article");
    article.className = "message message-bot";

    const content = document.createElement("div");
    content.className = "message-content";

    const label = createTextElement(
        "p",
        "ABC Tech Smart Bot",
        "message-label"
    );

    const answerText = createTextElement(
        "p",
        answer
    );

    content.appendChild(label);
    content.appendChild(answerText);

    if (Array.isArray(sources) && sources.length > 0) {
        const sourcesSection = createSourcesSection(sources);
        content.appendChild(sourcesSection);
    }

    article.appendChild(content);
    messageArea.appendChild(article);

    scrollMessagesToBottom();
}


/* ---------------------------------------------------------
   Source rendering
   --------------------------------------------------------- */

function createSourcesSection(sources) {
    const section = document.createElement("section");
    section.className = "sources";

    section.setAttribute(
        "aria-label",
        "Answer sources"
    );

    const heading = createTextElement(
        "h3",
        "Sources",
        "sources-heading"
    );

    const list = document.createElement("ul");
    list.className = "source-list";

    sources.forEach((source) => {
        const item = document.createElement("li");

        const citation = source.citation || "";

        const headingText =
            source.heading || "Company documentation";

        const sourceName =
            source.source || "Unknown source";

        const documentVersion =
            source.document_version
                ? ` (version ${source.document_version})`
                : "";

        const citationElement = createTextElement(
            "span",
            citation,
            "source-citation"
        );

        const description = createTextElement(
            "span",
            ` ${headingText} — ${sourceName}${documentVersion}`
        );

        item.appendChild(citationElement);
        item.appendChild(description);

        list.appendChild(item);
    });

    section.appendChild(heading);
    section.appendChild(list);

    return section;
}


/* ---------------------------------------------------------
   API request
   --------------------------------------------------------- */

async function askQuestion(question) {
    const response = await fetch(API_URL, {
        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            question: question
        })
    });

    let data = null;

    try {
        data = await response.json();
    } catch (error) {
        throw new Error(
            "The server returned an invalid response."
        );
    }

    if (!response.ok) {
        throw new Error(
            data?.error ||
            "The server could not process your question."
        );
    }

    return data;
}


/* ---------------------------------------------------------
   Form validation
   --------------------------------------------------------- */

function validateQuestion() {
    const question = questionInput.value.trim();

    if (!question) {
        showError(
            "Please enter a question before sending it."
        );

        questionInput.focus();

        return null;
    }

    if (question.length > 1000) {
        showError(
            "Your question is too long. Please keep it under 1000 characters."
        );

        questionInput.focus();

        return null;
    }

    return question;
}


/* ---------------------------------------------------------
   Form submission
   --------------------------------------------------------- */

async function handleSubmit(event) {
    event.preventDefault();

    clearError();

    const question = validateQuestion();

    if (!question) {
        return;
    }

    addUserMessage(question);

    questionInput.value = "";

    setLoading(true);

    try {
        const data = await askQuestion(question);

        const answer =
            data.answer ||
            "I couldn't find an answer in the available company documentation.";

        const sources =
            Array.isArray(data.sources)
                ? data.sources
                : [];

        addBotMessage(answer, sources);

    } catch (error) {
        console.error("Chat request failed:", error);

        showError(
            error.message ||
            "Something went wrong while contacting the assistant."
        );

        statusMessage.textContent =
            "The question could not be completed.";

    } finally {
        setLoading(false);

        /*
         * Return focus to the question field so keyboard users
         * can immediately ask another question.
         */
        questionInput.focus();
    }
}


/* ---------------------------------------------------------
   Keyboard shortcut
   --------------------------------------------------------- */

questionInput.addEventListener("keydown", (event) => {
    /*
     * Ctrl + Enter / Command + Enter submits the question.
     * Normal Enter continues to create a new line.
     */
    if (
        event.key === "Enter" &&
        (event.ctrlKey || event.metaKey)
    ) {
        event.preventDefault();

        chatForm.requestSubmit();
    }
});


/* ---------------------------------------------------------
   Form listener
   --------------------------------------------------------- */

chatForm.addEventListener(
    "submit",
    handleSubmit
);