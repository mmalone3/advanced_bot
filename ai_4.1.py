import speech_recognition as sr
import sys
import tkinter as tk
import pyttsx3
import pyjokes
import datetime
import wikipedia
import pyaudio 
import pywhatkit as pymus
import gradio
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import os
import json
import logging
from collections import deque
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from pydub import AudioSegment
import tempfile
import schedule
import openai
import asyncio

# Create a recognizer object
recognizer = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # 0 for male, 1 for female

# Initialize the greetings and goodbyes
hello = "how can i help you today"
goodbye = "goodbye"
action = ''

# API keys
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

TELEGRAM_BOT_TOKEN = "7259466291:AAFZtCWDuSze8Uj71AebwmNTVH6BGe5iMCY"
# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Initialize logger
# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Initialize other components
conversation_history = []
history_file = "conversation_history.json"
vectorizer = CountVectorizer()
lda_model = LatentDirichletAllocation(n_components=20, random_state=42)
sia = SentimentIntensityAnalyzer()
context_queue = deque(maxlen=10)  # Initialize context_queue with a maximum length of 10

# Load conversation history
if os.path.exists(history_file):
    with open(history_file, 'r') as f:
        conversation_history = json.load(f)

def load_conversation_history():
    global conversation_history
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            conversation_history = json.load(f)
    return conversation_history

def save_conversation_history():
    global conversation_history
    with open(history_file, 'w') as f:
        json.dump(conversation_history, f)
    return True

async def add_to_conversation_history(user_input, bot_response):
    global conversation_history
    new_entry = {"user_input": user_input, "bot_response": bot_response}
    conversation_history.append(new_entry)
    save_conversation_history()
    return new_entry

def perform_topic_modeling(messages):
    corpus = vectorizer.fit_transform([m["content"] for m in messages if "content" in m])
    lda_model.fit(corpus)
    topics = lda_model.transform(corpus)
    return topics.tolist()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

def periodic_retraining():
    global conversation_history
    print("Periodic retraining initiated...")
    # Call OpenAI's API to retrain the model
    return True

schedule.every().day.at("00:00").do(periodic_retraining)

def google_api():
    with sr.Microphone() as source:
        print("Listening... (Say 'exit' to quit)")
        audio = recognizer.listen(source)

    try:
        # Perform speech recognition
        text = recognizer.recognize_google(audio)
        text = text.lower()
        print(f"Recognized: {text}")
        return text

    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

def quit_app():
    root.destroy()  # Close the window

def interactions():
    text = google_api()
    
    if text is None:  # Add check for None
        return
    
    if "date" in text:
        date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        print(date)
        engine.say(date)
        engine.runAndWait()
        
    elif "joke" in text:
        joke = pyjokes.get_joke()
        print(joke)
        engine.say(joke)
        engine.runAndWait()
    elif "who is" in text:
        person = text.replace("who is", "")
        info = wikipedia.summary(person, 2)
        print(info)
        engine.say(info)
        engine.runAndWait()
    elif "play" in text:
        song = text.replace("play", "")
        print(f"Playing {song}")
        engine.say(f"Playing {song}")
        engine.runAndWait()
        pymus.playonyt(song)
        
    elif "exit" in text:
        print("Goodbye!")
        engine.say("Goodbye!")
        engine.runAndWait()
        sys.exit()
        
    elif "g radio" in text or "gradio" in text:
        print("Launching Gradio interface with ChatGPT integration...")
        engine.say("Launching Gradio interface with ChatGPT integration.")
        engine.runAndWait()
        launch_gradio_chatgpt()
        
    elif "telegram" in text:
        print("Starting Telegram bot...")
        engine.say("Starting Telegram bot.")
        engine.runAndWait()
        start_telegram_bot()
        
    else:
        print("I'm sorry, I did not understand that.")
        engine.say("I'm sorry, I did not understand that.")
        engine.runAndWait()
    
    return text

def launch_gradio_chatgpt():
    def chatbot(message, history):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                *history,
                {"role": "user", "content": message},
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return f"User: {message}\n\nAssistant: {response.choices[0].message.content}"
        except Exception as e:
            print(f"Error in chatbot function: {str(e)}")
            return f"An error occurred: {str(e)}"

    demo = gradio.ChatInterface(
        chatbot,
        title="AI Assistant Chatbot with ChatGPT",
        description="Chat with an AI-powered bot using ChatGPT"
    )
    demo.launch(share=True)
    print("Gradio interface with ChatGPT integration launched successfully!")
    print("Please open http://127.0.0.1:7860 in your browser to interact with the Gradio interface.")


    demo = gradio.ChatInterface(
        chatbot,
        title="AI Assistant Chatbot with ChatGPT",
        description="Chat with an AI-powered bot using ChatGPT"
    )
    demo.launch(share=True)
    print("Gradio interface with ChatGPT integration launched successfully!")
    print("Please open http://127.0.0.1:7860 in your browser to interact with the Gradio interface.")

def launch_gradio():
    def chatbot(message, history):
        return f"You said: {message}"

    demo = gradio.Interface(
        chatbot,
        inputs="text",
        outputs="text",
        title="AI Assistant Chatbot",
        description="Chat with an AI-powered bot"
    )
    demo.launch(share=True)
    print("Gradio interface launched successfully!")
    print("Please open http://127.0.0.1:7860 in your browser to interact with the Gradio interface.")

def start_telegram_bot():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    async def start(update: Update, context):
        await update.message.reply_text("Hello! I am an AI assistant. How can I help you today?")
        await update.message.reply_text("Please type your message below and I will respond to you as soon as possible.")

    async def text_message(update: Update, context):
        message = update.message.text
        if not message:
            return
        
        sentiment = analyze_sentiment(message)
        context_queue.append(message)
        context_str = " ".join(context_queue)
        
        try:
            topics = perform_topic_modeling(conversation_history + [{"role": "user", "content": message}])
        except KeyError as e:
            logger.error(f"KeyError: {e}")
            await update.message.reply_text("An error occurred while processing your message.")
            return
        
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Context: {context_str}. Sentiment: {sentiment}. Topic: {topics[0]}."},
            {"role": "user", "content": message}
        ]
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
        )
        await update.message.reply_text(response.choices[0].message.content)
        await add_to_conversation_history(message, response.choices[0].message.content)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))

    application.run_polling()

# Initialize the Tkinter root window
root = tk.Tk()
quit_button = tk.Button(root, text="Quit", command=quit_app)
quit_button.pack()

# Start the interactions
engine.say(hello)
engine.runAndWait()
done = interactions()
print(done)

# Start the Tkinter main loop
root.mainloop()
