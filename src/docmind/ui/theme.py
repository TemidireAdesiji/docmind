"""Custom dark-theme CSS for the Gradio interface."""

STYLESHEET = """
/* =========== layout =========== */
.progress-text { display: none !important; }

.gradio-container {
    max-width: 960px !important;
    width: 100% !important;
    margin: 0 auto !important;
    font-family:
        -apple-system, BlinkMacSystemFont,
        'Segoe UI', Roboto, sans-serif;
    background: #101010 !important;
}

/* =========== tabs =========== */
button[role="tab"] {
    color: #9e9e9e !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all .2s ease !important;
    background: transparent !important;
}
button[role="tab"]:hover { color: #e0e0e0 !important; }
button[role="tab"][aria-selected="true"] {
    color: #fff !important;
    border-bottom: 2px solid #fff !important;
    background: transparent !important;
}
.tabs          { border-bottom: none !important; }
.tab-nav       { border-bottom: 1px solid #333 !important; }
button[role="tab"]::before,
button[role="tab"]::after,
.tabs::before, .tabs::after,
.tab-nav::before, .tab-nav::after {
    display: none !important; content: none !important;
}
#docs-panel { max-width: 520px !important; margin: 0 auto !important; }

/* =========== buttons =========== */
button {
    border-radius: 8px !important;
    border: none !important;
    font-weight: 500 !important;
    transition: all .2s ease !important;
    box-shadow: none !important;
}
.primary {
    background: #2563eb !important;
    color: #fff !important;
}
.primary:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
}
.stop {
    background: #dc2626 !important;
    color: #fff !important;
}
.stop:hover {
    background: #b91c1c !important;
    transform: translateY(-1px) !important;
}

/* =========== chat input =========== */
textarea[placeholder="Type a message..."],
textarea[data-testid*="textbox"]:not(#kb-file-list textarea) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
textarea[placeholder="Type a message..."]:focus {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
form:has(textarea[placeholder="Type a message..."]) > div {
    background: transparent !important;
    border: none !important;
    gap: 10px !important;
}
form:has(textarea[placeholder="Type a message..."]) button {
    background: transparent !important;
    border: none !important;
    padding: 8px !important;
}
form:has(textarea[placeholder="Type a message..."]) button:hover {
    background: rgba(37, 99, 235, .1) !important;
}
form:has(textarea[placeholder="Type a message..."]) {
    gap: 10px !important; display: flex !important;
}

/* =========== file upload =========== */
.file-preview, [data-testid="file-upload"] {
    background: #181818 !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: #fff !important;
    min-height: 200px !important;
}
.file-preview:hover, [data-testid="file-upload"]:hover {
    border-color: #2563eb !important;
    background: #1c1c1c !important;
}
.file-preview *, [data-testid="file-upload"] * {
    color: #fff !important;
}
.file-preview .label, [data-testid="file-upload"] .label {
    display: none !important;
}

/* =========== inputs =========== */
input, textarea {
    background: #181818 !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
    transition: border-color .2s ease !important;
}
input:focus, textarea:focus {
    border-color: #2563eb !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, .12) !important;
}
textarea[readonly] {
    background: #181818 !important;
    color: #9e9e9e !important;
}

/* =========== file list =========== */
#kb-file-list {
    background: #181818 !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    padding: 10px !important;
}
#kb-file-list textarea {
    background: transparent !important;
    border: none !important;
    color: #e0e0e0 !important;
    padding: 0 !important;
}

/* =========== chatbot =========== */
.chatbot {
    border-radius: 6px !important;
    background: #181818 !important;
    border: none !important;
}
.chatbot .message-wrap, .chatbot > div {
    gap: 8px !important; padding: 12px !important;
}
.message { border-radius: 10px !important; }
.message.user {
    background: #2563eb !important;
    color: #fff !important;
}
.message.bot {
    background: #1e1e1e !important;
    color: #e0e0e0 !important;
    border: 1px solid #333 !important;
    width: fit-content !important;
    max-width: 90% !important;
}
.message-row img { margin: 0 !important; }
.avatar-container img { padding: 0 !important; }

/* =========== progress =========== */
.progress-bar-wrap {
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #181818 !important;
}
.progress-bar {
    border-radius: 10px !important;
    background: #2563eb !important;
}

/* =========== typography =========== */
h1, h2, h3, h4, h5, h6 { color: #e0e0e0 !important; }

/* =========== global =========== */
* { box-shadow: none !important; }
footer { visibility: hidden; }
"""
