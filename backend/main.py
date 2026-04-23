from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
import httpx
import json
import os
import re
import uuid
from typing import AsyncGenerator, List, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODELS_CACHE = []
LAST_FETCH = None

async def fetch_free_models():
    global MODELS_CACHE, LAST_FETCH
    if OPENROUTER_API_KEY:
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
                )
                if resp.status_code == 200:
                    all_models = resp.json().get("data", [])
                    free_models = []
                    for m in all_models:
                        pricing = m.get("pricing", {})
                        if pricing.get("prompt", 0) == 0:
                            free_models.append({
                                "id": m["id"],
                                "name": m.get("name", m["id"]),
                                "context_length": m.get("context_length", 8192),
                                "description": m.get("description", "")[:200]
                            })
                    MODELS_CACHE = free_models[:50]
                    LAST_FETCH = datetime.now()
            except:
                pass
    if not MODELS_CACHE:
        MODELS_CACHE = [
            {"id": "openrouter/stepfun/step-3.5-flash:free", "name": "Step 3.5 Flash", "context_length": 8192},
            {"id": "openrouter/deepseek/deepseek-r1-0528:free", "name": "DeepSeek R1", "context_length": 128000},
            {"id": "openrouter/openai/gpt-oss-120b:free", "name": "GPT-OSS 120B", "context_length": 8192}
        ]

@app.on_event("startup")
async def startup():
    await fetch_free_models()

@app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Free Claude Web</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                height: 100vh;
                overflow: hidden;
            }
            .container { display: flex; height: 100vh; }
            .sidebar {
                width: 280px;
                background: #1e1e1e;
                border-right: 1px solid #333;
                display: flex;
                flex-direction: column;
                overflow-y: auto;
            }
            .sidebar-header {
                padding: 20px;
                border-bottom: 1px solid #333;
                text-align: center;
            }
            .sidebar-header h2 { color: #4CAF50; margin-bottom: 10px; }
            .new-chat-btn {
                width: 100%;
                padding: 10px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }
            .chat-area { flex: 1; display: flex; flex-direction: column; }
            .messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
            }
            .message { margin-bottom: 20px; animation: fadeIn 0.3s ease; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            .message.user { text-align: right; }
            .message-content {
                display: inline-block;
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 12px;
                background: #2d2d2d;
            }
            .message.user .message-content { background: #4CAF50; color: white; }
            .message.assistant .message-content { background: #252526; border: 1px solid #333; }
            .input-area {
                border-top: 1px solid #333;
                padding: 20px;
                background: #1e1e1e;
            }
            .model-selector {
                margin-bottom: 12px;
            }
            .model-selector select {
                background: #2d2d2d;
                border: 1px solid #444;
                color: #e0e0e0;
                padding: 8px;
                border-radius: 6px;
                width: 100%;
            }
            #userInput {
                width: 100%;
                background: #2d2d2d;
                border: 1px solid #444;
                color: #e0e0e0;
                padding: 12px;
                border-radius: 8px;
                font-family: inherit;
                resize: vertical;
                font-size: 16px;
            }
            .send-btn {
                margin-top: 12px;
                padding: 10px 20px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                width: 100%;
                font-size: 16px;
            }
            .github-link {
                color: #4CAF50;
                text-decoration: none;
            }
            pre {
                background: #1e1e1e;
                padding: 12px;
                border-radius: 6px;
                overflow-x: auto;
                margin: 8px 0;
            }
            code {
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
            @media (max-width: 768px) {
                .sidebar { display: none; }
                .message-content { max-width: 95%; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="sidebar">
                <div class="sidebar-header">
                    <h2>🤖 Free Claude</h2>
                    <button class="new-chat-btn" onclick="location.reload()">+ New Chat</button>
                </div>
            </div>
            <div class="chat-area">
                <div class="messages" id="messages">
                    <div class="message assistant">
                        <div class="message-content">
                            👋 Hi! I'm Free Claude Web.<br><br>
                            I can:<br>
                            • Read any GitHub repository<br>
                            • Write and explain code<br>
                            • Answer programming questions<br><br>
                            <strong>Try pasting a GitHub URL!</strong>
                        </div>
                    </div>
                </div>
                <div class="input-area">
                    <div class="model-selector">
                        <select id="modelSelect"></select>
                    </div>
                    <textarea id="userInput" placeholder="Ask me anything... or paste a GitHub URL" rows="3"></textarea>
                    <button class="send-btn" onclick="sendMessage()">📤 Send</button>
                </div>
            </div>
        </div>
        <script>
            let currentMessages = [];
            let currentResponse = null;
            
            async function loadModels() {
                try {
                    const res = await fetch('/api/models');
                    const models = await res.json();
                    const select = document.getElementById('modelSelect');
                    select.innerHTML = '';
                    models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.id;
                        option.textContent = `${model.name} (${Math.round(model.context_length/1000)}k ctx)`;
                        select.appendChild(option);
                    });
                } catch(e) {
                    console.error('Failed to load models:', e);
                }
            }
            
            async function sendMessage() {
                const input = document.getElementById('userInput');
                const message = input.value.trim();
                if (!message) return;
                
                addMessage('user', message);
                currentMessages.push({ role: 'user', content: message });
                input.value = '';
                
                const model = document.getElementById('modelSelect').value;
                
                const loadingMsg = addMessage('assistant', '🤔 Thinking...');
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            messages: currentMessages,
                            model: model
                        })
                    });
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let fullResponse = '';
                    loadingMsg.remove();
                    const assistantMsg = addMessage('assistant', '');
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data !== '[DONE]') {
                                    try {
                                        const parsed = JSON.parse(data);
                                        fullResponse += parsed.content;
                                        updateMessage(assistantMsg, fullResponse);
                                    } catch(e) {}
                                }
                            }
                        }
                    }
                    
                    currentMessages.push({ role: 'assistant', content: fullResponse });
                    
                } catch(error) {
                    loadingMsg.remove();
                    addMessage('assistant', 'Error: ' + error.message);
                }
            }
            
            function addMessage(role, content) {
                const messagesDiv = document.getElementById('messages');
                const msgDiv = document.createElement('div');
                msgDiv.className = `message ${role}`;
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.innerHTML = renderMarkdown(content);
                msgDiv.appendChild(contentDiv);
                messagesDiv.appendChild(msgDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                return msgDiv;
            }
            
            function updateMessage(msgElement, content) {
                const contentDiv = msgElement.querySelector('.message-content');
                contentDiv.innerHTML = renderMarkdown(content);
            }
            
            function renderMarkdown(content) {
                let html = content.replace(/```(\\w+)?\\n([\\s\\S]*?)```/g, (match, lang, code) => {
                    return `<pre><code>${escapeHtml(code)}</code></pre>`;
                });
                html = html.replace(/https?:\\/\\/github\\.com\\/[\\w\\-\\.]+\\/[\\w\\-\\.]+/g, (url) => {
                    return `<a href="${url}" target="_blank" class="github-link">📦 ${url}</a>`;
                });
                html = html.replace(/\\n/g, '<br>');
                return html;
            }
            
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            document.getElementById('userInput').addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            loadModels();
        </script>
    </body>
    </html>
    """)

@app.get("/api/models")
async def get_models():
    await fetch_free_models()
    return MODELS_CACHE

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    model = data.get("model", "openrouter/stepfun/step-3.5-flash:free")
    
    # Check for GitHub URLs in messages
    last_message = messages[-1].get("content", "") if messages else ""
    github_urls = re.findall(r'https?://github\.com/[\w\-\.]+/[\w\-\.]+', last_message)
    
    # Enhance prompt if GitHub URL found
    if github_urls:
        system_msg = f"The user shared these GitHub repos: {', '.join(github_urls)}. Help them understand the code."
        messages.insert(0, {"role": "system", "content": system_msg})
    
    async def generate():
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": True,
                        "temperature": 0.7,
                        "max_tokens": 4000
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            if line == "data: [DONE]":
                                yield "data: [DONE]\n\n"
                            else:
                                yield line + "\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'content': f'Error: {str(e)}'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
