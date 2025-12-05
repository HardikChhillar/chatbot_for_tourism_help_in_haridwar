# app.py  — Improved Haridwar Assistant Flask app
from flask import Flask, render_template, request, jsonify
import json
import re
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string
import os
import pickle
from threading import Lock

# NOTE: In production, download required NLTK data during build/deploy (not at runtime).
# The try/except below is safe for local dev.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

app = Flask(__name__)

# ---------- Config / Files ----------
KNOWLEDGE_BASE_FILE = 'knowledge_base.json'
CONVERSATION_FILE = 'conversation_history.pkl'
LEARNING_DATA_FILE = 'learning_data.json'

file_lock = Lock()

# Load knowledge base safely
def load_knowledge_base(path=KNOWLEDGE_BASE_FILE):
    if not os.path.exists(path):
        print(f"[WARN] Knowledge base file not found at {path}. Using empty KB.")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load knowledge base: {e}")
        return {}

knowledge_base = load_knowledge_base()

# ---------- Conversation storage helpers ----------
def load_conversation_history():
    """Load conversation history from file"""
    if os.path.exists(CONVERSATION_FILE):
        try:
            with file_lock:
                with open(CONVERSATION_FILE, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load conversation history: {e}")
            return []
    return []

def save_conversation(user_query, bot_response, intent, entities):
    """Save conversation for learning (keeps last 1000)"""
    try:
        history = load_conversation_history()
        # Normalize bot_response to string for storage
        response_text = ''
        if isinstance(bot_response, dict):
            # try to get main 'response' key
            response_text = bot_response.get('response', '')
        else:
            response_text = str(bot_response)
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'bot_response': response_text,
            'intent': intent,
            'entities': entities,
            'response_type': getattr(bot_response, 'get', lambda k, d=None: d)('type', 'unknown') if isinstance(bot_response, dict) else 'unknown'
        }
        history.append(conversation)
        if len(history) > 1000:
            history = history[-1000:]
        with file_lock:
            with open(CONVERSATION_FILE, 'wb') as f:
                pickle.dump(history, f)
    except Exception as e:
        print(f"[WARN] Failed to save conversation: {e}")

# ---------- Learning data helpers ----------
def load_learning_data():
    """Load ML learning data"""
    if os.path.exists(LEARNING_DATA_FILE):
        try:
            with open(LEARNING_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load learning data: {e}")
            return {
                'intent_patterns': {},
                'entity_patterns': {},
                'common_queries': {},
                'user_preferences': {},
                'response_effectiveness': {}
            }
    return {
        'intent_patterns': {},
        'entity_patterns': {},
        'common_queries': {},
        'user_preferences': {},
        'response_effectiveness': {}
    }

def save_learning_data(learning_data):
    """Save ML learning data"""
    try:
        with open(LEARNING_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(learning_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to save learning data: {e}")

def analyze_conversations():
    """Analyze conversation history to improve responses"""
    history = load_conversation_history()
    if not history:
        return None
    learning_data = load_learning_data()
    # Intents
    intent_counts = Counter([c.get('intent', 'unknown') for c in history])
    learning_data['intent_patterns'] = dict(intent_counts.most_common(10))
    # Common queries (first 50 chars normalized)
    query_counts = Counter([c.get('user_query', '').lower()[:50] for c in history])
    learning_data['common_queries'] = dict(query_counts.most_common(20))
    # Entities
    entity_patterns = {}
    for conv in history:
        entities = conv.get('entities', {})
        for entity_type, entity_data in entities.items():
            if entity_type not in entity_patterns:
                entity_patterns[entity_type] = Counter()
            if isinstance(entity_data, dict):
                for key in entity_data.keys():
                    entity_patterns[entity_type][key] += 1
            elif isinstance(entity_data, list):
                for item in entity_data:
                    if isinstance(item, tuple) and len(item) > 0:
                        entity_patterns[entity_type][str(item[0])] += 1
    learning_data['entity_patterns'] = {k: dict(v.most_common(10)) for k, v in entity_patterns.items()}
    save_learning_data(learning_data)
    return learning_data

def generate_comprehensive_answer(user_input, intent, entities):
    """Generate comprehensive answer covering ALL aspects of user's need"""
    user_lower = user_input.lower()
    response_parts = []
    
    # Analyze what user really wants
    wants_list = []
    if any(word in user_lower for word in ['hospital', 'clinic', 'doctor', 'medical', 'emergency']):
        wants_list.append('medical')
    if any(word in user_lower for word in ['hotel', 'stay', 'accommodation', 'dharamshala']):
        wants_list.append('accommodation')
    if any(word in user_lower for word in ['restaurant', 'food', 'eat', 'cafe']):
        wants_list.append('food')
    if any(word in user_lower for word in ['shop', 'buy', 'shopping', 'market']):
        wants_list.append('shopping')
    if any(word in user_lower for word in ['temple', 'mandir', 'spiritual', 'ashram']):
        wants_list.append('spiritual')
    if any(word in user_lower for word in ['travel', 'agent', 'tour', 'package']):
        wants_list.append('travel')
    
    # If multiple needs detected, provide comprehensive answer
    if len(wants_list) > 1 or 'everything' in user_lower or 'all' in user_lower:
        response_parts.append("**Comprehensive Information for Your Needs:**\n\n")
        
        if 'medical' in wants_list or 'everything' in user_lower:
            hospitals = knowledge_base.get('hospitals_clinics', {})
            response_parts.append("**🏥 Medical Facilities:**\n")
            for pid, pdata in list(hospitals.items())[:3]:
                response_parts.append(f"• {pdata.get('name','')} 📞 {pdata.get('phone','N/A')} | {pdata.get('location','N/A')}\n")
            response_parts.append("\n")
        
        if 'accommodation' in wants_list or 'everything' in user_lower:
            hotels = knowledge_base.get('hotels_dharamshalas', {})
            response_parts.append("**🏨 Accommodation Options:**\n")
            for pid, pdata in list(hotels.items())[:3]:
                response_parts.append(f"• {pdata.get('name','')} 💰 {pdata.get('price_range','N/A')} | {pdata.get('location','N/A')}\n")
            response_parts.append("\n")
        
        if 'food' in wants_list or 'everything' in user_lower:
            restaurants = sorted(knowledge_base.get('cafes_restaurants', {}).items(),
                               key=lambda x: x[1].get('rating',0), reverse=True)
            response_parts.append("**🍽️ Top Restaurants:**\n")
            for pid, pdata in list(restaurants)[:3]:
                response_parts.append(f"• {pdata.get('name','')} ⭐ {pdata.get('rating','N/A')}/5.0 | {pdata.get('location','N/A')}\n")
            response_parts.append("\n")
        
        if 'shopping' in wants_list or 'everything' in user_lower:
            shops = knowledge_base.get('shopping_areas', {})
            response_parts.append("**🛍️ Shopping Areas:**\n")
            for pid, pdata in list(shops.items())[:2]:
                response_parts.append(f"• {pdata.get('name','')} | {pdata.get('location','N/A')}\n")
            response_parts.append("\n")
        
        if 'spiritual' in wants_list or 'everything' in user_lower:
            places = knowledge_base.get('places', {})
            response_parts.append("**🏛️ Major Temples:**\n")
            for pid, pdata in list(places.items())[:3]:
                response_parts.append(f"• {pdata.get('name','')}: {pdata.get('description','')}\n")
            response_parts.append("\n")
        
        if 'travel' in wants_list or 'everything' in user_lower:
            agents = knowledge_base.get('travel_agents', {})
            response_parts.append("**🚗 Travel Agents:**\n")
            for pid, pdata in list(agents.items())[:2]:
                response_parts.append(f"• {pdata.get('name','')} 📞 {pdata.get('phone','N/A')}\n")
            response_parts.append("\n")
        
        response_parts.append("💡 Ask me about any specific item for detailed information!")
        return ''.join(response_parts)
    
    return None

def improve_response_with_ml(user_input, intent, entities, base_response):
    """Use ML insights to possibly improve a base response (simple heuristics)"""
    learning_data = load_learning_data()
    user_lower50 = user_input.lower()[:50]
    common_queries = learning_data.get('common_queries', {})
    similar_queries = [q for q in common_queries.keys() if user_lower50 in q or q in user_lower50]
    # Example heuristic for itinerary queries
    if similar_queries and intent == 'itinerary' and isinstance(base_response, dict):
        if ('1' in user_input or 'one' in user_input.lower()) and 'day 2' in base_response.get('response', '').lower():
            # strip multi-day parts
            response_text = base_response.get('response', '')
            lines = response_text.split('\n')
            filtered_lines = []
            skip = False
            for line in lines:
                if re.search(r'\bday\s*2\b|\bday\s*3\b', line, re.IGNORECASE):
                    skip = True
                if not skip:
                    filtered_lines.append(line)
            base_response['response'] = '\n'.join(filtered_lines)
    return base_response

# ---------- NLP init ----------
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

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
    tokens = word_tokenize(text)
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and token not in string.punctuation:
            processed_tokens.append(lemmatizer.lemmatize(token))
    return ' '.join(processed_tokens), tokens

def classify_intent(user_input):
    user_lower = user_input.lower()
    intent_scores = {}
    # pattern scoring
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for patt in patterns:
            if re.search(patt, user_lower, re.IGNORECASE):
                score += 2
        intent_scores[intent] = score
    # simple keyword boost
    processed_text, _ = preprocess_text(user_input)
    keyword_intents = {
        'itinerary': ['itinerary', 'plan', 'schedule', 'route', 'tour', 'trip'],
        'place_info': ['tell', 'about', 'information', 'details', 'what is', 'where is'],
        'story': ['story', 'legend', 'mythology', 'history'],
        'help': ['help', 'emergency', 'contact', 'problem'],
        'cultural': ['culture', 'tradition', 'ritual', 'festival'],
        'time': ['time', 'when', 'timing', 'hours'],
        'how_to': ['how', 'reach', 'get to', 'go to'],
        'cost': ['cost', 'price', 'fee', 'ticket']
    }
    for intent, keywords in keyword_intents.items():
        for keyword in keywords:
            if keyword in processed_text or keyword in user_input.lower():
                intent_scores[intent] = intent_scores.get(intent, 0) + 1
    if intent_scores:
        best_intent = max(intent_scores, key=intent_scores.get)
        if intent_scores[best_intent] > 0:
            return best_intent
    return 'general'

# ---------- Entity extraction ----------
def extract_entities(user_input):
    """Extract entities from ALL knowledge base categories"""
    entities = {'places': [], 'time': {}, 'numbers': [], 'hospitals': [], 'hotels': [], 'akhadas': [], 
                'travel_agents': [], 'sweet_shops': [], 'gem_shops': [], 'book_shops': [], 
                'quiet_places': [], 'cultural_places': [], 'more_temples': []}
    user_lower = user_input.lower()
    
    # Extract from all categories
    categories = {
        'places': knowledge_base.get('places', {}),
        'hospitals': knowledge_base.get('hospitals_clinics', {}),
        'hotels': knowledge_base.get('hotels_dharamshalas', {}),
        'akhadas': knowledge_base.get('akhadas_ashrams', {}),
        'travel_agents': knowledge_base.get('travel_agents', {}),
        'sweet_shops': knowledge_base.get('sweet_shops', {}),
        'gem_shops': knowledge_base.get('gem_shops', {}),
        'book_shops': knowledge_base.get('holy_book_shops', {}),
        'quiet_places': knowledge_base.get('hidden_quiet_places', {}),
        'cultural_places': knowledge_base.get('cultural_places', {}),
        'more_temples': knowledge_base.get('more_temples', {})
    }
    
    for category, data in categories.items():
        for item_id, item_data in data.items():
            name = item_data.get('name', '').lower()
            keywords = [k.lower() for k in item_data.get('keywords', [])]
            if name and name in user_lower:
                entities[category].append((item_id, item_data, 10))
            else:
                for kw in keywords:
                    if kw and kw in user_lower:
                        entities[category].append((item_id, item_data, 5))
                        break
    # Time patterns - check in order of specificity (longer patterns first)
    time_patterns = {
        'half_day': r'\b(half\s*day|4\s*hours?|5\s*hours?)\b',
        'three_day': r'\b(three\s*days?|3\s*days?)\b',
        'two_day': r'\b(two\s*days?|2\s*days?|weekend|couple\s*of\s*days)\b',
        'one_day': r'\b(one\s*day|1\s*day|full\s*day|single\s*day|8\s*hours?)\b',
        'morning': r'\b(morning|sunrise|dawn)\b',
        'evening': r'\b(evening|sunset|dusk|night)\b',
        'afternoon': r'\b(afternoon|midday|noon)\b'
    }
    for key, patt in time_patterns.items():
        if re.search(patt, user_lower, re.IGNORECASE):
            entities['time'][key] = True
    # Numbers (e.g., '2 days', '3 hours') - extract explicitly
    numbers = re.findall(r'\b(\d+)\s*(day|days|hour|hours|minute|minutes)\b', user_lower)
    entities['numbers'] = numbers
    
    # Also check for explicit "two days" or "2 days" in the input
    if re.search(r'\b(two|2)\s*days?\b', user_lower, re.IGNORECASE):
        entities['time']['two_day'] = True
    if re.search(r'\b(three|3)\s*days?\b', user_lower, re.IGNORECASE):
        entities['time']['three_day'] = True
    if re.search(r'\b(one|1)\s*day\b', user_lower, re.IGNORECASE):
        entities['time']['one_day'] = True
    
    return entities

def find_exact_place_match(user_input, entities):
    if not entities.get('places'):
        return None
    entities['places'].sort(key=lambda x: x[2], reverse=True)
    place_id, place_data, score = entities['places'][0]
    if score >= 5:
        return (place_id, place_data)
    return None

# ---------- Place utilities ----------
def get_connected_places(place_id):
    places = knowledge_base.get('places', {})
    if place_id not in places:
        return []
    place = places[place_id]
    connections = place.get('connections', [])
    result = []
    for cid in connections:
        if cid in places:
            cp = places[cid]
            result.append({'name': cp.get('name', ''), 'description': cp.get('description', ''), 'id': cid})
    return result

def get_distance_info(from_place, to_place):
    distance_matrix = knowledge_base.get('distance_matrix', {})
    if from_place in distance_matrix and to_place in distance_matrix[from_place]:
        return distance_matrix[from_place][to_place]
    return None

def get_nearest_places(current_place, available_places, visited=None):
    if visited is None:
        visited = set()
    distances = []
    distance_matrix = knowledge_base.get('distance_matrix', {})
    if current_place not in distance_matrix:
        return []
    for pid in available_places:
        if pid != current_place and pid not in visited:
            dist_info = get_distance_info(current_place, pid)
            if dist_info:
                distances.append((pid, dist_info.get('distance_km', 0), dist_info))
    distances.sort(key=lambda x: x[1])
    return distances

def optimize_route(place_ids, start_place=None):
    if not place_ids:
        return []
    places_data = knowledge_base.get('places', {})
    if not start_place:
        start_place = 'har_ki_pauri' if 'har_ki_pauri' in place_ids or 'har_ki_pauri' in places_data else place_ids[0]
    if start_place not in place_ids:
        place_ids = [start_place] + place_ids
    optimized_route = []
    visited = set()
    current = start_place
    if current in places_data:
        optimized_route.append({
            'id': current,
            'name': places_data[current].get('name', ''),
            'visiting_time': places_data[current].get('visiting_time', ''),
            'best_time': places_data[current].get('best_time', ''),
            'description': places_data[current].get('description', ''),
            'travel_from_prev': None
        })
        visited.add(current)
    while len(visited) < len(place_ids):
        nearest = get_nearest_places(current, place_ids, visited)
        if not nearest:
            for pid in place_ids:
                if pid not in visited:
                    current = pid
                    break
            else:
                break
        else:
            next_place_id, distance, dist_info = nearest[0]
            place = places_data.get(next_place_id, {})
            optimized_route.append({
                'id': next_place_id,
                'name': place.get('name', ''),
                'visiting_time': place.get('visiting_time', ''),
                'best_time': place.get('best_time', ''),
                'description': place.get('description', ''),
                'travel_from_prev': {
                    'from': optimized_route[-1]['name'] if optimized_route else '',
                    'distance_km': distance,
                    'time_min': dist_info.get('auto_min', dist_info.get('walking_min', 0)),
                    'mode': dist_info.get('mode', 'auto')
                }
            })
            visited.add(next_place_id)
            current = next_place_id
    return optimized_route

def parse_visiting_time(time_str):
    """Return minutes. Attempts to handle many formats."""
    if not time_str:
        return 60
    s = str(time_str).lower()
    # explicit minutes/hours
    m = re.search(r'(\d+)\s*hour', s)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r'(\d+)\s*min', s)
    if m:
        return int(m.group(1))
    # ranges like 1-2 hours
    m = re.search(r'(\d+)\s*-\s*(\d+)\s*hour', s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) * 30
    # small heuristics
    if 'half' in s:
        return 45
    if 'short' in s:
        return 45
    return 60

# ---------- Itinerary generation ----------
def create_optimized_itinerary(time_info, entities, user_preferences=None):
    templates = knowledge_base.get('itinerary_templates', {})
    actual_days = 1  # Default
    
    # Debug: Check what we detected
    time_dict = entities.get('time', {}) if isinstance(entities, dict) else {}
    numbers = entities.get('numbers', []) if isinstance(entities, dict) else []
    
    # Strict template selection - CHECK IN ORDER: half_day, three_day, two_day, one_day
    # This order matters to avoid false matches
    # Check numbers array - it's list of tuples like [('2', 'days')]
    has_two_days = False
    has_three_days = False
    has_one_day = False
    
    for num_tuple in numbers:
        if len(num_tuple) >= 2:
            num_str = str(num_tuple[0])
            unit = str(num_tuple[1]).lower()
            if 'day' in unit:
                if num_str == '2' or num_str == 'two':
                    has_two_days = True
                elif num_str == '3' or num_str == 'three':
                    has_three_days = True
                elif num_str == '1' or num_str == 'one':
                    has_one_day = True
    
    if 'half_day' in time_info or time_dict.get('half_day'):
        template = templates.get('half_day', {})
        max_hours = 5
        max_places = 2
        actual_days = 0.5
    elif 'three_day' in time_info or time_dict.get('three_day') or has_three_days:
        template = templates.get('three_day', {})
        max_hours = 24
        max_places = 9
        actual_days = 3
    elif 'two_day' in time_info or time_dict.get('two_day') or has_two_days:
        template = templates.get('two_day', {})
        max_hours = 16
        max_places = 7
        actual_days = 2
    elif 'one_day' in time_info or time_dict.get('one_day') or has_one_day:
        template = templates.get('one_day', {})
        max_hours = 10
        max_places = 4
        actual_days = 1
    else:
        # Default to one day only if nothing specified
        template = templates.get('one_day', {})
        max_hours = 10
        max_places = 4
        actual_days = 1
    
    place_ids = template.get('places', [])[:max_places]
    optimized_route = optimize_route(place_ids)
    return optimized_route, template.get('description', ''), max_hours, actual_days

def format_optimized_itinerary_response(itinerary, description, max_hours, actual_days):
    """Format itinerary - actual_days is now passed directly, not parsed from description"""
    if not itinerary:
        return "I couldn't create an itinerary. Please specify how many days you have (e.g., '1 day', '2 days', 'half day')."
    
    # Format days text
    if actual_days == 0.5:
        days_text = "Half Day"
    elif actual_days == 1:
        days_text = "1 Day"
    elif actual_days == 2:
        days_text = "2 Days"
    elif actual_days == 3:
        days_text = "3 Days"
    else:
        days_text = f"{actual_days} Days"
    
    response = f"**Your Optimized {days_text} Itinerary for Haridwar:**\n\n"
    response += "📍 *Route optimized by distance for efficient travel*\n"
    response += f"⏱️ *Strictly limited to {actual_days} day(s) as requested*\n\n"
    total_time_minutes = 0
    day = 1
    day_start_time = 6 * 60
    current_time = day_start_time
    if actual_days == 0.5:
        places_per_day = len(itinerary)
    elif actual_days == 1:
        places_per_day = len(itinerary)
    elif actual_days == 2:
        places_per_day = (len(itinerary) + 1) // 2
    elif actual_days == 3:
        places_per_day = (len(itinerary) + 2) // 3
    else:
        places_per_day = len(itinerary) // max(1, int(actual_days))
    for i, item in enumerate(itinerary):
        if actual_days > 1 and i > 0 and i % places_per_day == 0 and day < actual_days:
            day += 1
            current_time = day_start_time
            response += f"\n---\n\n**Day {day}:**\n"
        elif i == 0:
            response += f"**Day {day}:**\n"
        visiting_min = parse_visiting_time(item.get('visiting_time', ''))
        travel_min = 0
        travel_info = ""
        if item.get('travel_from_prev'):
            travel = item['travel_from_prev']
            travel_min = travel.get('time_min', 0)
            travel_info = f"\n  🚗 Travel: {travel_min} min ({travel.get('distance_km', 0)} km) from {travel.get('from', '')} via {travel.get('mode', 'auto')}"
            current_time += travel_min
        hour = current_time // 60
        minute = current_time % 60
        time_str = f"{hour:02d}:{minute:02d}"
        response += f"\n**{time_str} - {item.get('name','')}**\n"
        response += f"  ⏱️ Duration: {item.get('visiting_time','')}\n"
        if travel_info:
            response += travel_info
        response += f"\n  📍 {item.get('description','')}\n"
        response += f"  ⭐ Best Time: {item.get('best_time','')}\n"
        current_time += visiting_min
        total_time_minutes += visiting_min + travel_min
    response += f"\n\n**Total Estimated Time:** {total_time_minutes // 60} hours {total_time_minutes % 60} minutes\n"
    response += "\n**💡 Pro Tips:**\n"
    response += "• Start early (6-7 AM) to avoid crowds and make the most of your day\n"
    response += "• Visit Har Ki Pauri twice: morning for holy dip, evening for Ganga Aarti (6-7 PM)\n"
    response += "• Nearby places are grouped together to minimize travel time\n"
    response += "• Carry water, comfortable shoes, and modest clothing for temple visits\n"
    response += "• Auto-rickshaws are readily available; negotiate fare before boarding\n"
    return response

# ---------- Clarifying questions ----------
def ask_clarifying_question(missing_info):
    questions = {
        'duration': "How many days do you have for your visit? (e.g., '1 day', '2 days', 'half day')",
        'start_location': "Where would you like to start your journey? (e.g., 'Har Ki Pauri', 'Railway Station', 'Hotel')",
        'preferences': "Do you have any specific preferences? (e.g., 'temples only', 'include wildlife', 'cultural sites')",
        'time_of_day': "What time of day would you prefer to start? (morning, afternoon, or flexible)"
    }
    return questions.get(missing_info, "Could you provide more details about your visit?")

# ---------- main response generator ----------
def get_response(user_input):
    user_input_lower = user_input.lower().strip()
    greetings = ['hi', 'hello', 'hey', 'namaste', 'good morning', 'good afternoon', 'good evening', 'greetings']
    if any(g in user_input_lower for g in greetings):
        return {
            'response': "Namaste! 🙏 Welcome to Haridwar Tourism Assistant. I can help with itineraries, place info, cultural stories, travel tips, and help contacts. What would you like to know?",
            'type': 'greeting'
        }
    entities = extract_entities(user_input)
    intent = classify_intent(user_input)
    # Itinerary
    if intent == 'itinerary':
        time_info = entities.get('time', {})
        numbers = entities.get('numbers', [])
        
        # Also check user input directly for "two days", "2 days", etc.
        user_lower = user_input.lower()
        if not time_info and not numbers:
            # Direct check in user input
            if re.search(r'\b(two|2)\s*days?\b', user_lower, re.IGNORECASE):
                time_info['two_day'] = True
                entities['numbers'] = [('2', 'days')]
            elif re.search(r'\b(three|3)\s*days?\b', user_lower, re.IGNORECASE):
                time_info['three_day'] = True
                entities['numbers'] = [('3', 'days')]
            elif re.search(r'\b(one|1)\s*day\b', user_lower, re.IGNORECASE):
                time_info['one_day'] = True
                entities['numbers'] = [('1', 'day')]
            elif re.search(r'\bhalf\s*day\b', user_lower, re.IGNORECASE):
                time_info['half_day'] = True
            else:
                return {'response': ask_clarifying_question('duration'), 'type': 'clarification', 'needs_info': 'duration'}
        
        itinerary, description, max_hours, actual_days = create_optimized_itinerary(time_info, entities)
        response_text = format_optimized_itinerary_response(itinerary, description, max_hours, actual_days)
        return {'response': response_text, 'type': 'itinerary'}
    # Place info
    if intent == 'place_info':
        place_match = find_exact_place_match(user_input, entities)
        if place_match:
            place_id, place_data = place_match
            query_type = 'full'
            if 'story' in user_input_lower or 'legend' in user_input_lower:
                query_type = 'story'
            elif 'time' in user_input_lower or 'when' in user_input_lower or 'hours' in user_input_lower:
                query_type = 'time'
            elif 'how' in user_input_lower or 'reach' in user_input_lower:
                query_type = 'how_to'
            elif 'cost' in user_input_lower or 'price' in user_input_lower or 'fee' in user_input_lower:
                query_type = 'cost'
            response_text = get_place_info(place_id, place_data, query_type)
            return {'response': response_text, 'type': 'place_detail'}
        else:
            places = knowledge_base.get('places', {})
            response_text = "Here are the major places to visit in Haridwar:\n\n"
            for pid, pdata in list(places.items())[:8]:
                response_text += f"• **{pdata.get('name','')}**: {pdata.get('description','')}\n"
            response_text += "\nAsk me about any specific place for detailed information!"
            return {'response': response_text, 'type': 'places_list'}
    # Story
    if intent == 'story':
        place_match = find_exact_place_match(user_input, entities)
        if place_match:
            place_id, place_data = place_match
            story = place_data.get('cultural_story', '')
            if story:
                response_text = f"**{place_data.get('name','')} - Cultural Story:**\n\n{story}\n"
                connected = get_connected_places(place_id)
                if connected:
                    response_text += "\n**Related Places:**\n"
                    for conn in connected[:2]:
                        cp = knowledge_base.get('places', {}).get(conn['id'], {})
                        if cp.get('cultural_story'):
                            response_text += f"\n**{conn['name']}**: {cp.get('cultural_story','')[:150]}...\n"
                return {'response': response_text, 'type': 'story'}
        return {'response': "Please specify which place you want the story of (e.g., 'story of Har Ki Pauri').", 'type': 'story'}
    # Help
    if intent == 'help':
        help_info = knowledge_base.get('help_contacts', [])
        response_text = "**Help & Support in Haridwar:**\n\n"
        for item in help_info:
            response_text += f"**{item.get('title','')}**\n{item.get('answer','')}\n\n"
        return {'response': response_text, 'type': 'help'}
    # Cultural
    if intent == 'cultural':
        cultural_info = knowledge_base.get('cultural_context', [])
        response_text = ""
        for item in cultural_info[:2]:
            response_text += f"**{item.get('question','Cultural Info')}**\n{item.get('answer','')}\n\n"
        return {'response': response_text, 'type': 'cultural'}
    # Time
    if intent == 'time':
        place_match = find_exact_place_match(user_input, entities)
        if place_match:
            place_id, place_data = place_match
            best_time = place_data.get('best_time', '')
            visiting_time = place_data.get('visiting_time', '')
            response_text = f"**{place_data.get('name','')} - Visiting Times:**\n\n"
            if best_time:
                response_text += f"Best Time: {best_time}\n"
            if visiting_time:
                response_text += f"Duration: {visiting_time}\n"
            return {'response': response_text, 'type': 'time'}
        else:
            tips = knowledge_base.get('travel_tips', [])
            for tip in tips:
                if 'best time' in tip.get('question', '').lower():
                    return {'response': tip.get('answer',''), 'type': 'time'}
    # How to (consolidated single branch)
    if intent == 'how_to':
        place_match = find_exact_place_match(user_input, entities)
        if place_match:
            place_id, place_data = place_match
            detailed_info = place_data.get('detailed_info', '')
            # Try to parse dedicated "How to Reach" block
            match = re.search(r'(How to Reach|how to reach)[^\n]*[:]\s*([^\n]+(?:\n[^\n]+)*)', detailed_info, re.IGNORECASE)
            if match:
                return {'response': f"**How to Reach {place_data.get('name','')}**\n\n{match.group(2)}", 'type': 'how_to'}
            # Fallback: distance from Har Ki Pauri
            dist_info = get_distance_info('har_ki_pauri', place_id)
            if dist_info:
                response_text = f"**How to Reach {place_data.get('name','')}:**\n\n"
                response_text += f"• Distance: {dist_info.get('distance_km','N/A')} km\n"
                response_text += f"• Travel Time: {dist_info.get('auto_min', dist_info.get('walking_min', 0))} minutes\n"
                response_text += f"• Mode: {dist_info.get('mode','auto')}\n"
                if dist_info.get('mode') == 'walking':
                    response_text += "\nYou can walk from Har Ki Pauri - it's very close!"
                elif dist_info.get('mode') == 'auto':
                    response_text += "\nTake an auto-rickshaw from Har Ki Pauri. Fare: ₹50-100 (negotiate)."
                else:
                    response_text += "\nTake a taxi or auto-rickshaw."
                return {'response': response_text, 'type': 'how_to'}
            return {'response': f"**{place_data.get('name','')}**\n\n{place_data.get('description','')}\n\nFor detailed directions, check the full info for this place.", 'type': 'how_to'}
    # Distance
    if intent == 'distance':
        places_mentioned = extract_entities(user_input).get('places', [])
        if len(places_mentioned) >= 2:
            p1_id, p1_data, _ = places_mentioned[0]
            p2_id, p2_data, _ = places_mentioned[1]
            dist_info = get_distance_info(p1_id, p2_id)
            if dist_info:
                response_text = f"**Distance from {p1_data.get('name','')} to {p2_data.get('name','')}:**\n\n"
                response_text += f"• Distance: {dist_info.get('distance_km','N/A')} km\n"
                response_text += f"• Travel Time: {dist_info.get('auto_min', dist_info.get('walking_min', 0))} minutes\n"
                response_text += f"• Recommended Mode: {dist_info.get('mode','auto')}\n"
                return {'response': response_text, 'type': 'distance'}
        elif len(places_mentioned) == 1:
            p_id, p_data, _ = places_mentioned[0]
            dist_info = get_distance_info('har_ki_pauri', p_id)
            if dist_info:
                response_text = f"**Distance to {p_data.get('name','')} from Har Ki Pauri:**\n\n"
                response_text += f"• Distance: {dist_info.get('distance_km','N/A')} km\n"
                response_text += f"• Travel Time: {dist_info.get('auto_min', dist_info.get('walking_min', 0))} minutes\n"
                response_text += f"• Recommended Mode: {dist_info.get('mode','auto')}\n"
                return {'response': response_text, 'type': 'distance'}
        return {'response': "Please mention the places (e.g., 'distance from Har Ki Pauri to Mansa Devi').", 'type': 'distance'}
    # Food
    if intent == 'food':
        cafes_restaurants = knowledge_base.get('cafes_restaurants', {})
        mentioned = []
        for pid, pdata in cafes_restaurants.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"💰 Price Range: {pdata.get('price_range','N/A')}\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"🕐 Timings: {pdata.get('timings','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            response_text += f"**Specialties:** {', '.join(pdata.get('specialties',[]))}\n\n"
            return {'response': response_text, 'type': 'food'}
        else:
            sorted_rest = sorted(cafes_restaurants.items(), key=lambda x: x[1].get('rating',0), reverse=True)
            response_text = "**Top Restaurants & Cafes in Haridwar:**\n\n"
            for pid, pdata in sorted_rest[:5]:
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}/5.0\n  {pdata.get('description','')}\n  {pdata.get('price_range','')} | {pdata.get('location','')}\n\n"
            response_text += "Ask me about any specific restaurant for details!"
            return {'response': response_text, 'type': 'food'}
    # Shopping
    if intent == 'shopping':
        shopping_areas = knowledge_base.get('shopping_areas', {})
        mentioned = []
        for pid, pdata in shopping_areas.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"🕐 Timings: {pdata.get('timings','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            response_text += f"**Items Available:** {', '.join(pdata.get('items_available',[]))}\n\n"
            return {'response': response_text, 'type': 'shopping'}
        else:
            response_text = "**Shopping Areas in Haridwar:**\n\n"
            for pid, pdata in shopping_areas.items():
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}\n  {pdata.get('description','')}\n  {pdata.get('location','')}\n\n"
            response_text += "Ask about a specific area for more details!"
            return {'response': response_text, 'type': 'shopping'}
    # Hospital/Clinic
    if intent == 'hospital':
        hospitals = knowledge_base.get('hospitals_clinics', {})
        mentioned = []
        for pid, pdata in hospitals.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"📞 Phone: {pdata.get('phone','N/A')}\n"
            response_text += f"🕐 Timings: {pdata.get('timings','N/A')}\n"
            response_text += f"📏 Distance: {pdata.get('distance_from_har_ki_pauri','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            response_text += f"**Services:** {', '.join(pdata.get('services',[]))}\n\n"
            response_text += f"**Best For:** {pdata.get('best_for','N/A')}\n"
            return {'response': response_text, 'type': 'hospital'}
        else:
            sorted_hospitals = sorted(hospitals.items(), key=lambda x: x[1].get('rating',0), reverse=True)
            response_text = "**Hospitals & Clinics in Haridwar:**\n\n"
            for pid, pdata in sorted_hospitals:
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}/5.0\n"
                response_text += f"  📞 {pdata.get('phone','N/A')} | 📍 {pdata.get('location','N/A')}\n"
                response_text += f"  {pdata.get('description','')}\n\n"
            response_text += "**Emergency:** Dial 102 or 108 for ambulance\n"
            response_text += "Ask me about any specific hospital for detailed information!"
            return {'response': response_text, 'type': 'hospital'}
    # Hotel/Dharamshala
    if intent == 'hotel':
        hotels = knowledge_base.get('hotels_dharamshalas', {})
        mentioned = []
        for pid, pdata in hotels.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"💰 Price: {pdata.get('price_range','N/A')}\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"📞 Phone: {pdata.get('phone','N/A')}\n"
            response_text += f"📏 Distance: {pdata.get('distance_from_har_ki_pauri','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            response_text += f"**Amenities:** {', '.join(pdata.get('amenities',[]))}\n\n"
            response_text += f"**Best For:** {pdata.get('best_for','N/A')}\n"
            return {'response': response_text, 'type': 'hotel'}
        else:
            sorted_hotels = sorted(hotels.items(), key=lambda x: x[1].get('rating',0), reverse=True)
            response_text = "**Hotels & Dharamshalas in Haridwar:**\n\n"
            for pid, pdata in sorted_hotels:
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}/5.0\n"
                response_text += f"  💰 {pdata.get('price_range','N/A')} | 📍 {pdata.get('location','N/A')}\n"
                response_text += f"  {pdata.get('description','')}\n\n"
            response_text += "Ask me about any specific hotel or dharamshala for details!"
            return {'response': response_text, 'type': 'hotel'}
    # Akhada/Ashram
    if intent == 'akhada':
        akhadas = knowledge_base.get('akhadas_ashrams', {})
        mentioned = []
        for pid, pdata in akhadas.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"📞 Phone: {pdata.get('phone','Contact through office')}\n"
            response_text += f"🕐 Timings: {pdata.get('timings','N/A')}\n"
            response_text += f"📏 Distance: {pdata.get('distance_from_har_ki_pauri','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            if pdata.get('significance'):
                response_text += f"**Significance:** {pdata.get('significance','')}\n\n"
            if pdata.get('programs'):
                response_text += f"**Programs:** {', '.join(pdata.get('programs',[]))}\n\n"
            if pdata.get('cultural_info'):
                response_text += f"**Cultural Info:** {pdata.get('cultural_info','')}\n\n"
            response_text += f"**Best For:** {pdata.get('best_for','N/A')}\n"
            return {'response': response_text, 'type': 'akhada'}
        else:
            response_text = "**Akhadas & Ashrams in Haridwar:**\n\n"
            for pid, pdata in akhadas.items():
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}/5.0\n"
                response_text += f"  📍 {pdata.get('location','N/A')}\n"
                response_text += f"  {pdata.get('description','')}\n\n"
            response_text += "Ask me about any specific akhada or ashram for detailed information!"
            return {'response': response_text, 'type': 'akhada'}
    # Travel Agent
    if intent == 'travel_agent':
        agents = knowledge_base.get('travel_agents', {})
        mentioned = []
        for pid, pdata in agents.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"📞 Phone: {pdata.get('phone','N/A')}\n"
            response_text += f"🕐 Timings: {pdata.get('timings','N/A')}\n"
            response_text += f"📏 Distance: {pdata.get('distance_from_har_ki_pauri','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            response_text += f"**Services:** {', '.join(pdata.get('services',[]))}\n\n"
            response_text += f"**Best For:** {pdata.get('best_for','N/A')}\n"
            return {'response': response_text, 'type': 'travel_agent'}
        else:
            response_text = "**Travel Agents & Tour Operators in Haridwar:**\n\n"
            for pid, pdata in agents.items():
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}/5.0\n"
                response_text += f"  📞 {pdata.get('phone','N/A')} | 📍 {pdata.get('location','N/A')}\n"
                response_text += f"  {pdata.get('description','')}\n\n"
            response_text += "Ask me about any specific travel agent for details!"
            return {'response': response_text, 'type': 'travel_agent'}
    # Sweet Shop
    if intent == 'sweet_shop':
        shops = knowledge_base.get('sweet_shops', {})
        mentioned = []
        for pid, pdata in shops.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"📞 Phone: {pdata.get('phone','N/A')}\n"
            response_text += f"🕐 Timings: {pdata.get('timings','N/A')}\n"
            response_text += f"📏 Distance: {pdata.get('distance_from_har_ki_pauri','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            response_text += f"**Specialties:** {', '.join(pdata.get('specialties',[]))}\n\n"
            response_text += f"**Best For:** {pdata.get('best_for','N/A')}\n"
            return {'response': response_text, 'type': 'sweet_shop'}
        else:
            sorted_shops = sorted(shops.items(), key=lambda x: x[1].get('rating',0), reverse=True)
            response_text = "**Sweet Shops in Haridwar:**\n\n"
            for pid, pdata in sorted_shops:
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}/5.0\n"
                response_text += f"  📍 {pdata.get('location','N/A')}\n"
                response_text += f"  {pdata.get('description','')}\n\n"
            response_text += "Ask me about any specific sweet shop for details!"
            return {'response': response_text, 'type': 'sweet_shop'}
    # Gem Shop
    if intent == 'gem_shop':
        shops = knowledge_base.get('gem_shops', {})
        mentioned = []
        for pid, pdata in shops.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"📞 Phone: {pdata.get('phone','N/A')}\n"
            response_text += f"🕐 Timings: {pdata.get('timings','N/A')}\n"
            response_text += f"📏 Distance: {pdata.get('distance_from_har_ki_pauri','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            response_text += f"**Items:** {', '.join(pdata.get('items',[]))}\n\n"
            response_text += f"**Best For:** {pdata.get('best_for','N/A')}\n"
            return {'response': response_text, 'type': 'gem_shop'}
        else:
            response_text = "**Gem Shops in Haridwar:**\n\n"
            for pid, pdata in shops.items():
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}/5.0\n"
                response_text += f"  📍 {pdata.get('location','N/A')}\n"
                response_text += f"  {pdata.get('description','')}\n\n"
            response_text += "Ask me about any specific gem shop for details!"
            return {'response': response_text, 'type': 'gem_shop'}
    # Book Shop
    if intent == 'book_shop':
        shops = knowledge_base.get('holy_book_shops', {})
        mentioned = []
        for pid, pdata in shops.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"📞 Phone: {pdata.get('phone','N/A')}\n"
            response_text += f"🕐 Timings: {pdata.get('timings','N/A')}\n"
            response_text += f"📏 Distance: {pdata.get('distance_from_har_ki_pauri','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            response_text += f"**Items:** {', '.join(pdata.get('items',[]))}\n\n"
            response_text += f"**Best For:** {pdata.get('best_for','N/A')}\n"
            return {'response': response_text, 'type': 'book_shop'}
        else:
            sorted_shops = sorted(shops.items(), key=lambda x: x[1].get('rating',0), reverse=True)
            response_text = "**Holy Book Shops in Haridwar:**\n\n"
            for pid, pdata in sorted_shops:
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}/5.0\n"
                response_text += f"  📍 {pdata.get('location','N/A')}\n"
                response_text += f"  {pdata.get('description','')}\n\n"
            response_text += "Ask me about any specific book shop for details!"
            return {'response': response_text, 'type': 'book_shop'}
    # Quiet Places
    if intent == 'quiet_place':
        places = knowledge_base.get('hidden_quiet_places', {})
        mentioned = []
        for pid, pdata in places.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"⏱️ Visiting Time: {pdata.get('visiting_time','N/A')}\n"
            response_text += f"⭐ Best Time: {pdata.get('best_time','N/A')}\n"
            response_text += f"📏 Distance: {pdata.get('distance_from_har_ki_pauri','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            response_text += f"**Why It's Quiet:** {pdata.get('why_quiet','N/A')}\n\n"
            response_text += f"**Best For:** {pdata.get('best_for','N/A')}\n"
            return {'response': response_text, 'type': 'quiet_place'}
        else:
            response_text = "**Quiet & Hidden Places in Haridwar:**\n\n"
            for pid, pdata in places.items():
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}/5.0\n"
                response_text += f"  📍 {pdata.get('location','N/A')}\n"
                response_text += f"  {pdata.get('description','')}\n"
                response_text += f"  Why Quiet: {pdata.get('why_quiet','N/A')}\n\n"
            response_text += "Ask me about any specific quiet place for details!"
            return {'response': response_text, 'type': 'quiet_place'}
    # Cultural Places
    if intent == 'cultural_place':
        places = knowledge_base.get('cultural_places', {})
        mentioned = []
        for pid, pdata in places.items():
            name = pdata.get('name','').lower()
            kws = [k.lower() for k in pdata.get('keywords',[])]
            if name in user_input_lower or any(kw in user_input_lower for kw in kws):
                mentioned.append((pid, pdata))
        if mentioned:
            pid, pdata = mentioned[0]
            response_text = f"**{pdata.get('name','')}**\n\n"
            response_text += f"⭐ Rating: {pdata.get('rating','N/A')}/5.0\n"
            response_text += f"📍 Location: {pdata.get('location','N/A')}\n"
            response_text += f"⏱️ Visiting Time: {pdata.get('visiting_time','N/A')}\n"
            response_text += f"⭐ Best Time: {pdata.get('best_time','N/A')}\n"
            response_text += f"💰 Entry Fee: {pdata.get('entry_fee','N/A')}\n"
            response_text += f"📏 Distance: {pdata.get('distance_from_har_ki_pauri','N/A')}\n\n"
            response_text += f"**Description:** {pdata.get('description','')}\n\n"
            response_text += f"**Best For:** {pdata.get('best_for','N/A')}\n"
            return {'response': response_text, 'type': 'cultural_place'}
        else:
            response_text = "**Cultural Places in Haridwar:**\n\n"
            for pid, pdata in places.items():
                response_text += f"• **{pdata.get('name','')}** ⭐ {pdata.get('rating','N/A')}/5.0\n"
                response_text += f"  📍 {pdata.get('location','N/A')}\n"
                response_text += f"  {pdata.get('description','')}\n\n"
            response_text += "Ask me about any specific cultural place for details!"
            return {'response': response_text, 'type': 'cultural_place'}
    # Cost
    if intent == 'cost':
        place_match = find_exact_place_match(user_input, extract_entities(user_input))
        if place_match:
            pid, pdata = place_match
            det = pdata.get('detailed_info','')
            match = re.search(r'(Entry Fee|entry fee|Fee)[^\n]*[:]\s*([^\n]+)', det, re.IGNORECASE)
            if match:
                return {'response': f"**Entry Fee for {pdata.get('name','')}:**\n\n{match.group(2)}", 'type': 'cost'}
            return {'response': f"**{pdata.get('name','')}**\n\nMost temples in Haridwar have free entry. Cable car rides cost around ₹150-200 per person.", 'type': 'cost'}
    # Comprehensive search across ALL categories for better matching
    all_entities = extract_entities(user_input)
    
    # Check all categories for matches
    for category in ['places', 'hospitals', 'hotels', 'akhadas', 'travel_agents', 'sweet_shops', 
                     'gem_shops', 'book_shops', 'quiet_places', 'cultural_places', 'more_temples']:
        if all_entities.get(category):
            items = all_entities[category]
            if items:
                items.sort(key=lambda x: x[2], reverse=True)
                item_id, item_data, score = items[0]
                if score >= 5:
                    # Generate comprehensive response
                    response_text = f"**{item_data.get('name','')}**\n\n"
                    response_text += f"**Description:** {item_data.get('description','')}\n\n"
                    
                    # Add all available information
                    if item_data.get('rating'):
                        response_text += f"⭐ Rating: {item_data.get('rating')}/5.0\n"
                    if item_data.get('location'):
                        response_text += f"📍 Location: {item_data.get('location')}\n"
                    if item_data.get('phone'):
                        response_text += f"📞 Phone: {item_data.get('phone')}\n"
                    if item_data.get('timings'):
                        response_text += f"🕐 Timings: {item_data.get('timings')}\n"
                    if item_data.get('price_range'):
                        response_text += f"💰 Price: {item_data.get('price_range')}\n"
                    if item_data.get('distance_from_har_ki_pauri'):
                        response_text += f"📏 Distance: {item_data.get('distance_from_har_ki_pauri')}\n"
                    if item_data.get('visiting_time'):
                        response_text += f"⏱️ Visiting Time: {item_data.get('visiting_time')}\n"
                    if item_data.get('best_time'):
                        response_text += f"⭐ Best Time: {item_data.get('best_time')}\n"
                    if item_data.get('entry_fee'):
                        response_text += f"💰 Entry Fee: {item_data.get('entry_fee')}\n"
                    
                    response_text += "\n"
                    
                    # Add category-specific details
                    if item_data.get('services'):
                        response_text += f"**Services:** {', '.join(item_data.get('services',[]))}\n\n"
                    if item_data.get('amenities'):
                        response_text += f"**Amenities:** {', '.join(item_data.get('amenities',[]))}\n\n"
                    if item_data.get('specialties'):
                        response_text += f"**Specialties:** {', '.join(item_data.get('specialties',[]))}\n\n"
                    if item_data.get('items'):
                        response_text += f"**Items:** {', '.join(item_data.get('items',[]))}\n\n"
                    if item_data.get('programs'):
                        response_text += f"**Programs:** {', '.join(item_data.get('programs',[]))}\n\n"
                    if item_data.get('cultural_info'):
                        response_text += f"**Cultural Information:** {item_data.get('cultural_info')}\n\n"
                    if item_data.get('why_quiet'):
                        response_text += f"**Why It's Quiet:** {item_data.get('why_quiet')}\n\n"
                    if item_data.get('best_for'):
                        response_text += f"**Best For:** {item_data.get('best_for')}\n"
                    
                    return {'response': response_text, 'type': category}
    
    # Fallback: place detail if match
    place_match = find_exact_place_match(user_input, all_entities)
    if place_match:
        pid, pdata = place_match
        return {'response': get_place_info(pid, pdata, 'full'), 'type': 'place_detail'}
    # Try comprehensive answer first
    comprehensive = generate_comprehensive_answer(user_input, intent, entities)
    if comprehensive:
        return {'response': comprehensive, 'type': 'comprehensive'}
    
    # Enhanced comprehensive search - understand user's complete need
    user_lower = user_input.lower()
    
    # Check for broad queries that need comprehensive answers
    comprehensive_keywords = {
        'everything': ['everything', 'all', 'complete', 'comprehensive', 'full information', 'all about'],
        'nearby': ['near', 'nearby', 'close to', 'around', 'surrounding'],
        'best': ['best', 'top', 'recommended', 'popular', 'famous'],
        'cheap': ['cheap', 'budget', 'affordable', 'low cost', 'economical'],
        'quiet': ['quiet', 'peaceful', 'less crowded', 'hidden', 'offbeat']
    }
    
    # If user asks for comprehensive info, provide it
    for key, keywords in comprehensive_keywords.items():
        if any(kw in user_lower for kw in keywords):
            if key == 'everything' and 'haridwar' in user_lower:
                # Provide comprehensive overview
                response_text = "**Complete Guide to Haridwar:**\n\n"
                response_text += "**🏛️ Major Places to Visit:**\n"
                places = knowledge_base.get('places', {})
                for pid, pdata in list(places.items())[:5]:
                    response_text += f"• {pdata.get('name','')}: {pdata.get('description','')}\n"
                response_text += "\n**🍽️ Top Restaurants:**\n"
                restaurants = knowledge_base.get('cafes_restaurants', {})
                for pid, pdata in list(sorted(restaurants.items(), key=lambda x: x[1].get('rating',0), reverse=True))[:3]:
                    response_text += f"• {pdata.get('name','')} ⭐ {pdata.get('rating','N/A')}/5.0\n"
                response_text += "\n**🏥 Medical Facilities:**\n"
                hospitals = knowledge_base.get('hospitals_clinics', {})
                for pid, pdata in list(hospitals.items())[:3]:
                    response_text += f"• {pdata.get('name','')} 📞 {pdata.get('phone','N/A')}\n"
                response_text += "\n**🏨 Accommodation:**\n"
                hotels = knowledge_base.get('hotels_dharamshalas', {})
                for pid, pdata in list(hotels.items())[:3]:
                    response_text += f"• {pdata.get('name','')} 💰 {pdata.get('price_range','N/A')}\n"
                response_text += "\n**🛍️ Shopping:** Main Bazaar (near Har Ki Pauri) for religious items and souvenirs\n"
                response_text += "\n**📚 Holy Books:** Gita Press Book Shop for religious texts\n"
                response_text += "\n**🍬 Sweets:** Madhur Milap for traditional sweets and prasad\n"
                response_text += "\nAsk me about any specific category for detailed information!"
                return {'response': response_text, 'type': 'comprehensive'}
            
            elif key == 'nearby' or key == 'near':
                # Find what they're asking near to
                base_place = None
                for pid, pdata in knowledge_base.get('places', {}).items():
                    if pdata.get('name','').lower() in user_lower:
                        base_place = pid
                        break
                if not base_place:
                    base_place = 'har_ki_pauri'  # Default
                
                response_text = f"**Places Near {knowledge_base.get('places', {}).get(base_place, {}).get('name', 'Har Ki Pauri')}:**\n\n"
                
                # Find nearby places from all categories
                distance_matrix = knowledge_base.get('distance_matrix', {}).get(base_place, {})
                nearby = []
                for pid, dist_info in distance_matrix.items():
                    if dist_info.get('distance_km', 999) <= 2.0:  # Within 2 km
                        nearby.append((pid, dist_info))
                nearby.sort(key=lambda x: x[1].get('distance_km', 999))
                
                for pid, dist_info in nearby[:5]:
                    # Search in all categories
                    found = False
                    for category in ['places', 'cafes_restaurants', 'hotels_dharamshalas', 'hospitals_clinics']:
                        items = knowledge_base.get(category, {})
                        if pid in items:
                            item = items[pid]
                            response_text += f"• **{item.get('name','')}** ({dist_info.get('distance_km',0)} km)\n"
                            response_text += f"  {item.get('description','')}\n\n"
                            found = True
                            break
                
                return {'response': response_text, 'type': 'nearby'}
            
            elif key == 'best':
                # Provide best rated items
                response_text = "**Best Rated Places in Haridwar:**\n\n"
                
                # Best restaurants
                restaurants = sorted(knowledge_base.get('cafes_restaurants', {}).items(), 
                                   key=lambda x: x[1].get('rating',0), reverse=True)
                response_text += "**🍽️ Top Restaurants:**\n"
                for pid, pdata in restaurants[:3]:
                    response_text += f"• {pdata.get('name','')} ⭐ {pdata.get('rating','N/A')}/5.0\n"
                
                # Best hotels
                hotels = sorted(knowledge_base.get('hotels_dharamshalas', {}).items(),
                              key=lambda x: x[1].get('rating',0), reverse=True)
                response_text += "\n**🏨 Top Hotels:**\n"
                for pid, pdata in hotels[:3]:
                    response_text += f"• {pdata.get('name','')} ⭐ {pdata.get('rating','N/A')}/5.0\n"
                
                # Best hospitals
                hospitals = sorted(knowledge_base.get('hospitals_clinics', {}).items(),
                                 key=lambda x: x[1].get('rating',0), reverse=True)
                response_text += "\n**🏥 Top Hospitals:**\n"
                for pid, pdata in hospitals[:3]:
                    response_text += f"• {pdata.get('name','')} ⭐ {pdata.get('rating','N/A')}/5.0\n"
                
                return {'response': response_text, 'type': 'best'}
    
    # Final fallback with comprehensive options
    places = knowledge_base.get('places', {})
    place_names = [p.get('name','') for p in list(places.values())[:5]]
    response_text = "I can help you with comprehensive information about Haridwar! Here's what I can assist with:\n\n"
    response_text += "**📍 Places:** Temples, attractions, hidden places, quiet spots\n"
    response_text += "**🍽️ Food:** Restaurants, cafes, sweet shops with ratings\n"
    response_text += "**🏨 Stay:** Hotels, dharamshalas, ashrams with prices\n"
    response_text += "**🏥 Medical:** Hospitals, clinics with contact numbers\n"
    response_text += "**🛍️ Shopping:** Markets, gem shops, book shops\n"
    response_text += "**🧘 Spiritual:** Akhadas, ashrams, spiritual centers\n"
    response_text += "**🚗 Travel:** Travel agents, tour packages\n"
    response_text += "**📅 Itinerary:** Customized plans based on your time\n\n"
    response_text += "**Popular Places:**\n"
    for name in place_names:
        response_text += f"• {name}\n"
    response_text += "\n**Examples:**\n"
    response_text += "• 'Everything about Haridwar' - Complete guide\n"
    response_text += "• 'Best restaurants' - Top rated places\n"
    response_text += "• 'Hotels near Har Ki Pauri' - Nearby accommodation\n"
    response_text += "• 'Quiet places' - Peaceful spots\n"
    response_text += "\nWhat specific information would you like?"
    return {'response': response_text, 'type': 'default'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = (data.get('message') or '').strip()
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        entities = extract_entities(user_message)
        intent = classify_intent(user_message)
        bot_response = get_response(user_message)
        bot_response = improve_response_with_ml(user_message, intent, entities, bot_response)
        save_conversation(user_message, bot_response, intent, entities)
        history = load_conversation_history()
        # analyze every 10th conversation (non-blocking simple approach)
        if len(history) % 10 == 0:
            try:
                analyze_conversations()
            except Exception as e:
                print(f"[WARN] analyze_conversations failed: {e}")
        return jsonify({
            'response': bot_response.get('response') if isinstance(bot_response, dict) else str(bot_response),
            'type': bot_response.get('type') if isinstance(bot_response, dict) else 'unknown',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        print(f"[ERROR] /chat exception: {e}")
        return jsonify({'error': str(e)}), 500

# Helper used above in get_response (kept here)
def get_place_info(place_id, place_data, query_type='full'):
    """Get specific information about a place based on query type"""
    response_parts = []
    if query_type in ('full','info'):
        response_parts.append(place_data.get('detailed_info', place_data.get('description', '')))
    if query_type in ('full','story'):
        story = place_data.get('cultural_story','')
        if story:
            response_parts.append(f"\n\n**Cultural Story:**\n{story}")
    if query_type in ('full','time'):
        best_time = place_data.get('best_time','')
        visiting_time = place_data.get('visiting_time','')
        if best_time or visiting_time:
            response_parts.append(f"\n\n**Visiting Information:**\n")
            if best_time:
                response_parts.append(f"Best Time: {best_time}\n")
            if visiting_time:
                response_parts.append(f"Duration: {visiting_time}\n")
    if query_type in ('full','how_to'):
        detailed_info = place_data.get('detailed_info','')
        match = re.search(r'(How to Reach|how to reach)[^\n]*\n([^\n]+)', detailed_info, re.IGNORECASE)
        if match:
            response_parts.append(f"\n\n**How to Reach:**\n{match.group(2)}")
    if query_type in ('full','cost'):
        detailed_info = place_data.get('detailed_info','')
        match = re.search(r'(Entry Fee|entry fee|Fee)[^\n]*[:]\s*([^\n]+)', detailed_info, re.IGNORECASE)
        if match:
            response_parts.append(f"\n\n**Entry Fee:**\n{match.group(2)}")
    if query_type == 'full':
        connected = get_connected_places(place_id)
        if connected:
            response_parts.append("\n\n**Connected Places You Might Like:**\n")
            for conn in connected[:3]:
                response_parts.append(f"• **{conn['name']}**: {conn['description']}\n")
    return ''.join(response_parts) if response_parts else place_data.get('description','')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
