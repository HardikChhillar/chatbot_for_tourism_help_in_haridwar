# Haridwar Tourism Assistant Chatbot

An intelligent and comprehensive chatbot designed to help tourists visiting Haridwar with personalized itinerary planning, detailed place information, cultural context, and assistance during their stay.

## Features

- **🤖 Intelligent Itinerary Planning**: Create personalized itineraries based on your time constraints (half-day, 1-day, 2-day, 3-day trips)
- **📍 Detailed Place Information**: Comprehensive details about each attraction including:
  - Historical significance and cultural context
  - Best time to visit and visiting duration
  - Entry fees and how to reach
  - What to experience at each place
  - Cultural stories and legends
  - Connected places and their relationships
- **📖 Cultural Stories & Connections**: Discover the interconnected stories, legends, and mythology behind places
- **🎯 Contextual Responses**: The chatbot understands your queries and provides relevant, detailed answers
- **🏛️ Tourist Attractions**: Information about major places including Har Ki Pauri, Mansa Devi Temple, Chandi Devi Temple, Daksh Mahadev Temple, Bharat Mata Mandir, Sapt Rishi Ashram, Rajaji National Park, and Gurukul Kangri University
- **🌍 Cultural Context**: Detailed information about the cultural significance, traditions, rituals, and festivals of Haridwar
- **🆘 Help & Support**: Emergency contacts, medical facilities, tourist information centers, and assistance for inconveniences
- **💡 Travel Tips**: Best time to visit, safety tips, and practical information for tourists

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd chatbot_for_tourism_help_in_haridwar
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: The first time you run the application, NLTK will automatically download required data (punkt, stopwords, wordnet, averaged_perceptron_tagger). This is a one-time setup and happens automatically.

## Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your web browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Start chatting!** The chatbot is ready to help you with:
   - **Itinerary Planning**: "Create itinerary for one day" or "I have 2 days, what should I see?"
   - **Place Details**: "Tell me about Har Ki Pauri" or "What is Mansa Devi Temple?"
   - **Cultural Stories**: "What is the story behind Chandi Devi?" or "How are the temples connected?"
   - **Cultural Information**: Traditions, rituals, and significance
   - **Emergency Contacts**: Help services and support
   - **Travel Tips**: Best time to visit, safety tips, and recommendations

## Usage

### Example Questions You Can Ask:

**Itinerary Planning:**
- "Create itinerary for one day"
- "I have 2 days in Haridwar, what should I see?"
- "Plan a trip for 3 days"
- "What can I see in half a day?"
- The chatbot will create an optimized route based on distances

**Distance Queries:**
- "How far is Mansa Devi from Har Ki Pauri?"
- "Distance between Chandi Devi and Bharat Mata"
- "How long to reach Daksh Mahadev from Har Ki Pauri?"

**Food & Dining:**
- "Best restaurants in Haridwar"
- "Where to eat near Har Ki Pauri"
- "Tell me about Chotiwala Restaurant"
- "Cafes with good ratings"
- "Sweet shops for prasad"

**Medical & Health:**
- "Hospitals in Haridwar"
- "Clinics near Har Ki Pauri"
- "Emergency medical facilities"
- "Ayurvedic treatment centers"

**Accommodation:**
- "Hotels in Haridwar"
- "Dharamshalas for pilgrims"
- "Budget accommodation"
- "Hotels near Har Ki Pauri"

**Spiritual:**
- "Akhadas in Haridwar"
- "Ashrams for yoga"
- "Spiritual centers"
- "Where to meet sadhus"

**Shopping:**
- "Gem shops for Rudraksha"
- "Holy book shops"
- "Where to buy prasad"
- "Shopping areas"

**Hidden & Quiet Places:**
- "Quiet places in Haridwar"
- "Less crowded spots"
- "Peaceful places for meditation"
- "Hidden attractions"

**Comprehensive Queries:**
- "Everything about Haridwar"
- "Complete guide to Haridwar"
- "Best places in Haridwar"
- "Nearby places to Har Ki Pauri"

**Shopping:**
- "Shopping areas in Haridwar"
- "Where to buy souvenirs"
- "Main Bazaar information"

**Place Information:**
- "Tell me about Har Ki Pauri"
- "What is Mansa Devi Temple?"
- "Details about Chandi Devi Temple"
- "Information about Daksh Mahadev Temple"

**Cultural Stories & Connections:**
- "What is the story behind Har Ki Pauri?"
- "How are Mansa Devi and Chandi Devi connected?"
- "Tell me the legend of Sapt Rishi Ashram"
- "What is the cultural significance of Haridwar?"

**General Queries:**
- "Best time to visit Haridwar?"
- "Emergency contacts in Haridwar"
- "What are the traditions and rituals?"
- "Medical help in Haridwar"
- "Transportation in Haridwar"

### Quick Question Buttons

The interface includes quick question buttons for:
- 1 Day Itinerary
- 2 Day Itinerary
- Har Ki Pauri
- Cultural Info
- Emergency Help
- Travel Tips

## Project Structure

```
chatbot_for_tourism_help_in_haridwar/
├── app.py                 # Flask backend application
├── knowledge_base.json    # Knowledge base with all information
├── templates/
│   └── index.html        # Frontend chat interface
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML, CSS, JavaScript
- **NLP**: NLTK (Natural Language Toolkit) for advanced text processing
- **Data Storage**: JSON-based knowledge base

## Features of the Chatbot

1. **Advanced NLP Processing**: 
   - Intent classification using pattern matching and keyword analysis
   - Named Entity Recognition (NER) for places, times, and numbers
   - Text preprocessing with tokenization, lemmatization, and stopword removal
   - Contextual query understanding for precise responses

2. **Distance-Based Route Optimization**: 
   - Complete distance and travel time matrix between all places
   - Nearest neighbor algorithm for optimal route planning
   - Automatically groups nearby places together
   - Shows travel times, distances, and recommended transport modes
   - Creates efficient itineraries that minimize travel time

3. **Intelligent Itinerary Planning**: 
   - Automatically creates personalized itineraries based on time constraints
   - Optimizes routes based on proximity (nearest places visited together)
   - Includes travel times and distances between places
   - Suggests best transport modes (walking, auto-rickshaw, taxi)
   - Time-stamped itinerary with realistic scheduling

4. **Conversational Flow**: 
   - Asks clarifying questions when information is missing
   - Understands context and builds on previous conversations
   - Provides direct, useful answers tailored to user needs

5. **Precise Query Understanding**: 
   - Understands exact user intent (itinerary, place info, story, help, cultural, time, how-to, cost, distance)
   - Provides direct answers instead of related information
   - Extracts specific entities (places, time, numbers) from queries
   - Can answer distance queries between any two places

4. **Detailed Place Information**: Each place includes:
   - Historical and cultural significance
   - Best visiting times and duration
   - Entry fees and transportation
   - What to experience
   - Cultural stories and legends
   - Connected places

5. **Contextual Responses**: 
   - Answers specific questions (time, cost, how-to) directly
   - Provides full information when requested
   - Adapts response based on query type

6. **Story Connections**: Discover how places are connected through history, mythology, and culture

7. **Time-Based Recommendations**: Provides suggestions based on morning, afternoon, or evening preferences

8. **Modern UI**: Beautiful, responsive chat interface with smooth animations

9. **Massive Comprehensive Knowledge Base** (960+ lines): 
   - **9 major attractions** with detailed information
   - **7 restaurants and cafes** with ratings and reviews
   - **3 shopping areas** with item listings
   - **5 hospitals and clinics** with contact numbers and services
   - **6 hotels and dharamshalas** with prices and amenities
   - **5 akhadas and ashrams** with spiritual programs
   - **3 travel agents** with services and packages
   - **3 sweet shops** with specialties
   - **2 gem shops** for Rudraksha and precious stones
   - **3 holy book shops** for religious texts
   - **4 hidden/quiet places** for peaceful visits
   - **4 more temples** (Vaishno Devi, Pawan Dham, etc.)
   - **3 cultural places** (museums, heritage sites)
   - Complete distance and travel time matrix
   - All with ratings, locations, timings, and detailed information

10. **Machine Learning & Learning System**:
    - Stores all conversations for analysis
    - Learns from user interactions to improve responses
    - Analyzes common query patterns
    - Adapts responses based on user behavior
    - Continuously improves accuracy over time

11. **Smart Itinerary Logic**:
    - **Respects exact time constraints** - 1 day = 1 day, not 5 days
    - Strictly limits places based on requested duration
    - No over-promising or under-delivering

12. **Easy to Extend**: Knowledge base is in JSON format, easy to update and expand

## Customization

You can easily customize the chatbot by editing `knowledge_base.json`. The structure includes:

**Places (`places`)**: Detailed information about each attraction
- `name`: Place name
- `keywords`: Keywords for matching queries
- `description`: Short description
- `detailed_info`: Comprehensive information
- `visiting_time`: How long to spend
- `best_time`: Best time of day to visit
- `cultural_story`: Cultural/historical story
- `connections`: Array of connected place IDs

**Location Data (`location_data`)**: Geographic information
- `coordinates`: Latitude and longitude
- `area`: Area name

**Distance Matrix (`distance_matrix`)**: Travel information between places
- Distance in kilometers
- Travel time in minutes (walking, auto-rickshaw)
- Recommended transport mode (walking, auto, taxi)
- Used for route optimization

**Cafes & Restaurants (`cafes_restaurants`)**: Dining options
- Restaurant/cafe name and type
- Rating (out of 5.0)
- Price range
- Location and distance from Har Ki Pauri
- Specialties and timings
- Best for recommendations

**Shopping Areas (`shopping_areas`)**: Shopping locations
- Market name and type
- Rating
- Location and distance
- Items available
- Timings and best for recommendations

## Machine Learning Features

The chatbot learns from every conversation:
- **Conversation Storage**: All chats are saved in `conversation_history.pkl`
- **Pattern Analysis**: Analyzes common queries and intent patterns
- **Response Improvement**: Uses learned patterns to improve future responses
- **Entity Recognition**: Learns common entity patterns from user queries
- **Automatic Learning**: Analyzes conversations every 10 interactions

**Learning Data**: Stored in `learning_data.json` with:
- Intent patterns (most common intents)
- Entity patterns (common entities extracted)
- Common queries (frequently asked questions)
- Response effectiveness tracking

**Itinerary Templates (`itinerary_templates`)**: Pre-defined itineraries
- `half_day`, `one_day`, `two_day`, `three_day`: Different duration templates
- Each template includes: `duration`, `places` (array of place IDs), `description`

**Cultural Context (`cultural_context`)**: Cultural information
- `question`: The question it answers
- `keywords`: List of keywords for matching
- `answer`: The response text

**Help Contacts (`help_contacts`)**: Help and emergency information
- `title`: Contact category
- `keywords`: Keywords for matching
- `answer`: Contact information

**Travel Tips (`travel_tips`)**: Travel advice
- `question`: The question it answers
- `keywords`: List of keywords for matching
- `answer`: The response text

## Notes

- The chatbot runs on `localhost:5000` by default
- Make sure port 5000 is available or modify the port in `app.py`
- The application is set to debug mode for development
- **Conversation History**: The chatbot automatically saves conversations in `conversation_history.pkl` for learning
- **Learning Data**: ML insights are stored in `learning_data.json`
- **Time Constraints**: The chatbot strictly respects time constraints - if you ask for 1 day, you get exactly 1 day itinerary
- **First Run**: NLTK will download required data on first run (automatic)

## Support

For issues or questions about the chatbot, please check the code comments or modify the knowledge base as needed.

---

**Namaste! 🙏 Enjoy your visit to Haridwar!**

