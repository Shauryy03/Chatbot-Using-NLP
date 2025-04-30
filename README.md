# 🤖 Chatbot Using NLP

A Python-based chatbot that uses Natural Language Processing (NLP) to understand and respond to user queries. The bot is trained on an intents.json dataset and uses a neural network (TensorFlow & TFLearn) for intent classification.

# ✨ Features
 
       ✅ Interactive Web UI built with Streamlit
       ✅ Natural Language Understanding using NLTK
       ✅ Neural Network Model for intent classification
       ✅ Easy Customization through intents.json
       ✅ Interactive chat interface in the terminal
       ✅ Real-time Chat Experience
       
# 🛠️ Tech Stack
 
       Python (Primary Language)
       NLTK (Natural Language Toolkit)
       TensorFlow (Deep Learning Framework)
       TFLearn (High-level TensorFlow API)
       NumPy (Numerical Computing) 

# 🚀 Implementation

     The chatbot(input_text) function takes user input, transforms it using the already fitted vectorizer, predicts the intent using the already trained clf model, 
     and then randomly selects a response associated with that intent from the intents data. B. Machine Learning (ML) Component:

       
# 🚀 Installation & Setup

      1.Repository
           https://github.com/Shauryy03/Chatbot-Using-NLP
      2. Install Dependencies
             pip install nltk tensorflow tflearn numpy streamlit
             (Includes streamlit, nltk, tensorflow, numpy)
      3. Download NLTK Data
            import nltk
            nltk.download('punkt')
      4. Train the Model
            python train.py
            (Generates model.tflearn and words.pkl.)
      5. Run the Chatbot
            streamlit run app.py
            ➡️ Opens automatically at http://localhost:8501
            
# Example Output:
 
      User: Hi  
      Bot: Hello! How can I help you?





