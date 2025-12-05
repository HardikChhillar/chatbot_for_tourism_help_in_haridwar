# app.py  — Improved Haridwar Assistant Flask app
import os, json, pickle, re, string
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from threading import Lock
from app_enhanced import handle_itinerary_request

# Detect Vercel serverless
ON_VERCEL = bool(os.getenv("VERCEL") or os.getenv("VERCEL_URL"))

# NLTK setup: avoid downloads on Vercel
if not ON_VERCEL:
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt', quiet=True)
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords', quiet=True)
    try: nltk.data.find('corpora/wordnet')
    except LookupError: nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# ---------- Config / Files ----------
KNOWLEDGE_BASE_FILE = 'knowledge_base.json'

# Use ephemeral /tmp on serverless instead of read-only project dir
DATA_DIR = '/tmp' if ON_VERCEL else '.'
CONVERSATION_FILE = os.path.join(DATA_DIR, 'conversation_history.pkl')
LEARNING_DATA_FILE = os.path.join(DATA_DIR, 'learning_data.json')

file_lock = Lock()

# Load knowledge base safely
def load_knowledge_base(path=KNOWLEDGE_BASE_FILE):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

knowledge_base = load_knowledge_base()

# ---------- Conversation storage helpers ----------
def load_conversation_history():
    """Load conversation history from file"""
    try:
        with file_lock:
            if os.path.exists(CONVERSATION_FILE):
                with open(CONVERSATION_FILE, 'rb') as f:
                    return pickle.load(f)
    except Exception:
        pass
    return []

def save_conversation(user_query, bot_response, intent, entities):
    """Save conversation for learning (keeps last 1000). Works on /tmp in serverless."""
    try:
        history = load_conversation_history()
        history.append({
            'ts': datetime.utcnow().isoformat(),
            'q': user_query, 'a': bot_response,
            'intent': intent, 'entities': entities
        })
        history = history[-1000:]
        with file_lock:
            with open(CONVERSATION_FILE, 'wb') as f:
                pickle.dump(history, f)
    except Exception:
        pass

def load_learning_data():
    """Load ML learning data"""
    try:
        with open(LEARNING_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'intent_patterns': {}, 'entity_patterns': {}, 'common_queries': {}}

def save_learning_data(learning_data):
    """Save ML learning data (to /tmp on Vercel)."""
    try:
        with open(LEARNING_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(learning_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def analyze_conversations():
    """Analyze conversation history to improve responses"""
    history = load_conversation_history()
    learning = load_learning_data()
    for item in history[-100:]:
        q = (item.get('q') or '').lower().strip()
        learning['common_queries'][q] = learning['common_queries'].get(q, 0) + 1
        intent = item.get('intent') or 'general'
        learning['intent_patterns'][intent] = learning['intent_patterns'].get(intent, 0) + 1
    save_learning_data(learning)
    return learning

# ---------- NLP init ----------
lemmatizer = WordNetLemmatizer()

# Safe tokenizers/stopwords when NLTK resources aren’t present
def _safe_tokens(text):
    try:
        return word_tokenize(text)
    except Exception:
        return re.findall(r'\b\w+\b', text)

try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {'the','is','in','at','to','for','a','an','and','of','on','with','this','that','it','you','i'}

# Replace usage inside preprocess_text
def preprocess_text(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = _safe_tokens(text)
    filtered = []
    for t in tokens:
        if t not in stop_words and t not in string.punctuation:
            try: filtered.append(lemmatizer.lemmatize(t))
            except Exception: filtered.append(t)
    return ' '.join(filtered), tokens

# ---------- INTENT PATTERNS ----------
INTENT_PATTERNS = {
    'itinerary': [
        r'\b(itinerary|plan|schedule|route|tour|trip|visit|see|explore|go)\b.*\b(day|hour|time|duration|long|weekend)\b',
        r'\b(create|make|suggest|recommend).*\b(itinerary|plan|schedule)\b'
    ],
    'place_info': [
        r'\b(tell|about|information|details|what is|where is|describe)\b',
    ],
    'story': [r'\b(story|legend|mythology|history|tale|narrative)\b'],
    'help': [r'\b(help|emergency|contact|problem|issue|support|assistance|inconvenience|trouble)\b'],
    'cultural': [r'\b(culture|cultural|tradition|ritual|festival|significance|importance)\b'],
    'time': [r'\b(best time|when to|what time|timing|hours|open|close)\b'],
    'how_to': [r'\b(how to|how do|how can|way to|reach|get to|go to)\b'],
    'cost': [r'\b(cost|price|fee|ticket|charge|expensive|cheap|free)\b'],
    'distance': [r'\b(distance|far|how far|km|kilometer|miles)\b'],
    'food': [r'\b(food|eat|restaurant|cafe|dining|meal|hungry|breakfast|lunch|dinner)\b'],
    'shopping': [r'\b(shopping|buy|shop|market|bazaar|souvenir|purchase)\b'],
    'hospital': [r'\b(hospital|clinic|doctor|medical|emergency|health|treatment|medicine)\b'],
    'hotel': [r'\b(hotel|accommodation|stay|lodging|room|dharamshala|ashram stay)\b'],
    'akhada': [r'\b(akhada|ashram|sadhu|spiritual center|yoga|meditation)\b'],
    'travel_agent': [r'\b(travel agent|tour operator|package|tour|yatra|char dham)\b'],
    'sweet_shop': [r'\b(sweet|mithai|prasad|sweet shop|jalebi|laddu)\b'],
    'gem_shop': [r'\b(gem|rudraksha|precious stone|jewelry|beads|mala)\b'],
    'book_shop': [r'\b(book|holy book|geeta|ramayana|religious book|spiritual book)\b'],
    'quiet_place': [r'\b(quiet|peaceful|hidden|offbeat|less crowded|meditation spot)\b'],
    'cultural_place': [r'\b(cultural|museum|heritage|history|traditional|artistic)\b']
}

def preprocess_text(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = _safe_tokens(text)
    filtered = []
    for t in tokens:
        if t not in stop_words and t not in string.punctuation:
            try: filtered.append(lemmatizer.lemmatize(t))
            except Exception: filtered.append(t)
    return ' '.join(filtered), tokens

def detect_intent(user_input):
    s, _ = preprocess_text(user_input)
    if re.search(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen)\s*day', s):
        return 'itinerary'
    if 'food' in s or 'street' in s: return 'food'
    if 'ashram' in s or 'yoga' in s or 'satsang' in s: return 'ashram'
    if 'day trip' in s or 'rishikesh' in s or 'mussoorie' in s: return 'day_trip'
    return 'general'

def get_response(user_input):
    intent = detect_intent(user_input)
    if intent == 'itinerary':
        enhanced = handle_itinerary_request(user_input)
        return enhanced.get('response', 'Itinerary ready.')
    if intent == 'food':
        return "Popular street food: Hoshiyar Puri (breakfast), Jalebi Point (Main Bazaar), Ganga Chaat Corner (railway area), Upper Road pakoras, Kankhal lassi."
    if intent == 'ashram':
        return "Ashram programs: morning aarti, yoga, meditation, scripture study, seva, evening satsang. Long stays: Maa Anandamayi (Kankhal), Shri Ganga Bhawan."
    if intent == 'day_trip':
        return "Day trips: Rishikesh (Ram/Lakshman Jhula, Beatles Ashram, cafes, rafting seasonal) and Mussoorie (Kempty Falls, Mall Road, Gun Hill, Camel’s Back)."
    return "Ask me for itineraries, street food, ashram routines, or day trips."

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True) or {}
    msg = (data.get('message') or '').strip()
    if not msg:
        return jsonify({'error':'Empty message'}), 400
    resp = get_response(msg)
    save_conversation(msg, resp, detect_intent(msg), {})
    return jsonify({'response': resp})

@app.route('/analytics', methods=['GET'])
def analytics():
    learning = analyze_conversations()
    history = load_conversation_history()
    return jsonify({'summary': {'total_conversations': len(history)}, 'learning_data': learning, 'recent_history': history[-50:]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
