import logging

from flask import Flask, request, jsonify
from agent import Service
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources=r'/api/*')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_service = Service()

@app.route('/')
def home():
    return "App Works!"

@app.route('/api/chat', methods=['POST'])
def chat_completions():
    try:
        content = request.json
        messages = content.get("messages", None)
        if messages is None:
            return jsonify({"error": "The 'messages' field is required."}), 400
        tools = content.get("tools", None)
        tool_choice = content.get("tool_choice", None)
        if tools is not None and tool_choice is None:
            content["tool_choice"] = "auto"
        response = api_service.chat_completions(**content)
        return jsonify(response), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500



