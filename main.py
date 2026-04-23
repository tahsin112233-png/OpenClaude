from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
import json
import os
import re
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

# CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY", "")

# Cache for models
all_models_cache = []
last_fetch_time = None
cache_duration = timedelta(hours=1)

async def fetch_all_free_models():
    """Fetch all free models from OpenRouter"""
    global all_models_cache, last_fetch_time
    
    # Return cached if fresh
    if last_fetch_time and datetime.now() - last_fetch_time < cache_duration and all_models_cache:
        return all_models_cache
    
    all_models = []
    
    # Fetch from OpenRouter if API key exists
    if OPENROUTER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    
                    for model in models:
                        pricing = model.get("pricing", {})
                        prompt_price = pricing.get("prompt", 1)
                        completion_price = pricing.get("completion", 1)
                        
                        # Check if it's free or has free tier
                        is_free = (prompt_price == 0 and completion_price == 0) or "free" in model.get("id", "").lower()
                        
                        if is_free:
                            all_models.append({
                                "id": model["id"],
                                "name": model.get("name", model["id"]),
                                "provider": model.get("provider", {}).get("name", "Unknown"),
                                "context_length": model.get("context_length", 8192),
                                "description": model.get("description", "")[:200],
                                "pricing": "Free",
                                "type": "openrouter"
                            })
        except Exception as e:
            print(f"Error fetching OpenRouter models: {e}")
    
    # Add NVIDIA NIM free models if API key exists
    if NVIDIA_NIM_API_KEY:
        nvidia_models = [
            {"id": "nvidia/nemotron-4-340b-instruct", "name": "Nemotron 4 340B", "provider": "NVIDIA", "context_length": 4096, "pricing": "40 req/min free", "type": "nvidia"},
            {"id": "nvidia/llama-3.1-nemotron-70b-instruct", "name": "Llama 3.1 Nemotron 70B", "provider": "NVIDIA", "context_length": 8192, "pricing": "40 req/min free", "type": "nvidia"},
            {"id": "nvidia/mistral-nemo-12b-instruct", "name": "Mistral NeMo 12B", "provider": "NVIDIA", "context_length": 8192, "pricing": "40 req/min free", "type": "nvidia"}
        ]
        all_models.extend(nvidia_models)
    
    # Add fallback free models if no API keys
    if not all_models:
        all_models = [
            {"id": "openrouter/stepfun/step-3.5-flash:free", "name": "Step 3.5 Flash", "provider": "StepFun", "context_length": 8192, "pricing": "Free", "type": "openrouter"},
            {"id": "openrouter/deepseek/deepseek-r1-0528:free", "name": "DeepSeek R1", "provider": "DeepSeek", "context_length": 128000, "pricing": "Free", "type": "openrouter"},
            {"id": "openrouter/openai/gpt-oss-120b:free", "name": "GPT-OSS 120B", "provider": "OpenAI", "context_length": 8192, "pricing": "Free", "type": "openrouter"},
            {"id": "openrouter/microsoft/phi-3-medium-128k:free", "name": "Phi-3 Medium", "provider": "Microsoft", "context_length": 128000, "pricing": "Free", "type": "openrouter"},
            {"id": "openrouter/google/gemini-2-flash-thinking-exp:free", "name": "Gemini 2 Flash", "provider": "Google", "context_length": 1048576, "pricing": "Free", "type": "openrouter"},
            {"id": "openrouter/meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B", "provider": "Meta", "context_length": 128000, "pricing": "Free", "type": "openrouter"}
        ]
    
    all_models_cache = all_models
    last_fetch_time = datetime.now()
    return all_models

# HTML Interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>Free Claude Code - AI Assistant with GitHub Intelligence</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }
        
        .app {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        
        /* Header */
        .header {
            background: #1e1e1e;
            border-bottom: 1px solid #333;
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .logo h1 {
            font-size: 1.3em;
            color: #4CAF50;
        }
        
        .badge {
            background: #4CAF50;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
        }
        
        .model-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .model-selector label {
            font-size: 12px;
            color: #888;
        }
        
        select {
            background: #2d2d2d;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            max-width: 250px;
        }
        
        .refresh-btn {
            background: #2d2d2d;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
        }
        
        .refresh-btn:hover {
            background: #3d3d3d;
        }
        
        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        /* Messages Area */
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            text-align: right;
        }
        
        .message-content {
            display: inline-block;
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 12px;
            background: #2d2d2d;
            text-align: left;
        }
        
        .message.user .message-content {
            background: #4CAF50;
            color: white;
        }
        
        .message.assistant .message-content {
            background: #252526;
            border: 1px solid #333;
        }
        
        .message.system .message-content {
            background: #1a3a1a;
            border: 1px solid #4CAF50;
            font-size: 12px;
        }
        
        /* Code blocks */
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
        
        .github-link {
            color: #4CAF50;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        
        /* Input Area */
        .input-area {
            border-top: 1px solid #333;
            padding: 16px;
            background: #1e1e1e;
        }
        
        textarea {
            width: 100%;
            background: #2d2d2d;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 12px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 16px;
            resize: vertical;
        }
        
        textarea:focus {
            outline: none;
            border-color: #4CAF50;
        }
        
        .input-actions {
            display: flex;
            gap: 10px;
            margin-top: 12px;
        }
        
        .send-btn {
            flex: 1;
            padding: 10px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
        }
        
        .send-btn:disabled {
            background: #666;
            cursor: not-allowed;
        }
        
        .clear-btn {
            padding: 10px 20px;
            background: #d32f2f;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        
        /* Loading indicator */
        .typing-indicator {
            display: inline-flex;
            gap: 4px;
            padding: 8px 12px;
        }
        
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #888;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                align-items: stretch;
            }
            .model-selector {
                justify-content: space-between;
            }
            select {
                flex: 1;
                max-width: none;
            }
            .message-content {
                max-width: 95%;
            }
        }
        
        /* Status */
        .status {
            font-size: 11px;
            color: #4CAF50;
            margin-top: 8px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <div class="logo">
                <h1>🤖 Free Claude Code</h1>
                <span class="badge">Free • Open Source • GitHub Ready</span>
            </div>
            <div class="model-selector">
                <label>🤯 Model:</label>
                <select id="modelSelect"></select>
                <button class="refresh-btn" onclick="refreshModels()">🔄 Refresh</button>
            </div>
        </div>
        
        <div class="main-content">
            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="message-content">
                        <strong>👋 Welcome to Free Claude Code!</strong><br><br>
                        I'm a free AI assistant that can:<br>
                        • 📖 <strong>Read any GitHub repository</strong> - just paste a URL!<br>
                        • 💻 <strong>Write and explain code</strong> in any language<br>
                        • 🔍 <strong>Analyze project structure</strong> and dependencies<br>
                        • 🆓 <strong>Completely free</strong> - no credit card needed<br><br>
                        <strong>Try these examples:</strong><br>
                        • "Explain this repo: https://github.com/facebook/react"<br>
                        • "Write a Python function to sort a list"<br>
                        • "What's the difference between React and Vue?"<br>
                        • "Create a simple HTML/CSS button with hover effect"<br>
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <textarea 
                    id="userInput" 
                    placeholder="Ask me anything... or paste a GitHub URL to analyze it!"
                    rows="3"
                ></textarea>
                <div class="input-actions">
                    <button class="send-btn" onclick="sendMessage()">📤 Send Message</button>
                    <button class="clear-btn" onclick="clearChat()">🗑️ Clear</button>
                </div>
                <div class="status" id="status">✅ Ready • Using OpenRouter API</div>
            </div>
        </div>
    </div>
    
    <script>
        let currentMessages = [];
        let currentAbortController = null;
        let currentModel = null;
        
        // Load models on startup
        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                if (!response.ok) throw new Error('Failed to load models');
                const models = await response.json();
                
                const select = document.getElementById('modelSelect');
                select.innerHTML = '';
                
                if (models.length === 0) {
                    select.innerHTML = '<option>No models available</option>';
                    return;
                }
                
                // Group by provider
                const grouped = {};
                models.forEach(model => {
                    const provider = model.provider || model.type || 'Other';
                    if (!grouped[provider]) grouped[provider] = [];
                    grouped[provider].push(model);
                });
                
                for (const [provider, providerModels] of Object.entries(grouped)) {
                    const group = document.createElement('optgroup');
                    group.label = provider;
                    
                    providerModels.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.id;
                        let ctx = Math.round(model.context_length / 1000);
                        option.textContent = `${model.name} (${ctx}k ctx) - ${model.pricing}`;
                        group.appendChild(option);
                    });
                    
                    select.appendChild(group);
                }
                
                if (models.length > 0) {
                    currentModel = models[0].id;
                }
                
                updateStatus(`✅ Loaded ${models.length} free models`);
            } catch (error) {
                console.error('Failed to load models:', error);
                updateStatus('⚠️ Using fallback models');
            }
        }
        
        async function refreshModels() {
            updateStatus('🔄 Refreshing models...');
            await loadModels();
        }
        
        function updateStatus(message) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            setTimeout(() => {
                if (statusDiv.textContent === message) {
                    statusDiv.textContent = '✅ Ready • Using OpenRouter API';
                }
            }, 3000);
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Disable send button
            const sendBtn = document.querySelector('.send-btn');
            sendBtn.disabled = true;
            
            // Add user message
            addMessage('user', message);
            currentMessages.push({ role: 'user', content: message });
            input.value = '';
            
            // Get selected model
            const modelSelect = document.getElementById('modelSelect');
            const model = modelSelect.value;
            
            // Add loading indicator
            const loadingId = addLoadingIndicator();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: currentMessages,
                        model: model
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantMessage = '';
                let messageElement = null;
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') continue;
                            
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.content) {
                                    assistantMessage += parsed.content;
                                    
                                    if (!messageElement) {
                                        removeLoadingIndicator(loadingId);
                                        messageElement = addMessage('assistant', assistantMessage);
                                    } else {
                                        updateMessage(messageElement, assistantMessage);
                                    }
                                }
                            } catch (e) {
                                // Ignore parse errors
                            }
                        }
                    }
                }
                
                if (assistantMessage) {
                    currentMessages.push({ role: 'assistant', content: assistantMessage });
                }
                
            } catch (error) {
                console.error('Error:', error);
                removeLoadingIndicator(loadingId);
                addMessage('assistant', `❌ Error: ${error.message}\\n\\nPlease make sure you've set your OpenRouter API key in Render environment variables.`);
            } finally {
                sendBtn.disabled = false;
                input.focus();
            }
        }
        
        function addMessage(role, content) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = renderMarkdown(content);
            
            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            return messageDiv;
        }
        
        function updateMessage(messageElement, content) {
            const contentDiv = messageElement.querySelector('.message-content');
            contentDiv.innerHTML = renderMarkdown(content);
        }
        
        function addLoadingIndicator() {
            const messagesDiv = document.getElementById('messages');
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message assistant';
            loadingDiv.id = 'loading-indicator';
            loadingDiv.innerHTML = `
                <div class="message-content">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            `;
            messagesDiv.appendChild(loadingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return loadingDiv;
        }
        
        function removeLoadingIndicator(element) {
            if (element && element.remove) {
                element.remove();
            }
        }
        
        function renderMarkdown(content) {
            let html = content;
            
            // Code blocks
            html = html.replace(/```(\\w+)?\\n([\\s\\S]*?)```/g, (match, lang, code) => {
                return `<pre><code>${escapeHtml(code)}</code></pre>`;
            });
            
            // Inline code
            html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
            
            // GitHub links
            html = html.replace(/https?:\\/\\/github\\.com\\/[\\w\\-\\.]+\\/[\\w\\-\\.]+/g, (url) => {
                return `<a href="${url}" target="_blank" class="github-link">📦 ${url}</a>`;
            });
            
            // Line breaks
            html = html.replace(/\\n/g, '<br>');
            
            return html;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function clearChat() {
            currentMessages = [];
            const messagesDiv = document.getElementById('messages');
            messagesDiv.innerHTML = `
                <div class="message assistant">
                    <div class="message-content">
                        <strong>🧹 Chat cleared!</strong><br><br>
                        Ready for new questions. Paste a GitHub URL to get started!
                    </div>
                </div>
            `;
        }
        
        // Enter to send
        document.getElementById('userInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Load models on startup
        loadModels();
        
        // Auto-refresh models every hour
        setInterval(loadModels, 3600000);
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    return HTMLResponse(HTML_TEMPLATE)

@app.get("/api/models")
async def get_models():
    models = await fetch_all_free_models()
    return models

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])
        model = data.get("model", "openrouter/stepfun/step-3.5-flash:free")
        
        # Detect GitHub URLs in last message
        if messages:
            last_content = messages[-1].get("content", "")
            github_urls = re.findall(r'https?://github\.com/[\w\-\.]+/[\w\-\.]+', last_content)
            
            if github_urls:
                # Add system message about GitHub
                github_context = f"The user shared these GitHub repositories: {', '.join(github_urls)}. Provide detailed analysis of the repository structure, main technologies, and how to work with it."
                messages.insert(0, {"role": "system", "content": github_context})
        
        async def generate():
            try:
                # Use OpenRouter if API key exists
                if OPENROUTER_API_KEY and "openrouter" in model:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        async with client.stream(
                            "POST",
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                                "Content-Type": "application/json",
                                "HTTP-Referer": "https://free-claude-web.onrender.com",
                                "X-Title": "Free Claude Web"
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
                elif NVIDIA_NIM_API_KEY and "nvidia" in model:
                    # NVIDIA NIM endpoint
                    actual_model = model.replace("nvidia/", "")
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        async with client.stream(
                            "POST",
                            "https://integrate.api.nvidia.com/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {NVIDIA_NIM_API_KEY}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": actual_model,
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
                else:
                    # Fallback response
                    yield f"data: {json.dumps({'content': '⚠️ No API key configured. Please set OPENROUTER_API_KEY in environment variables.\\n\\nGet your free key at: https://openrouter.ai/keys'})}\n\n"
                    yield "data: [DONE]\n\n"
                    
            except Exception as e:
                error_msg = f"❌ API Error: {str(e)}\\n\\nPlease check your OpenRouter API key is valid."
                yield f"data: {json.dumps({'content': error_msg})}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
