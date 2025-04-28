🤖 Chatbot Using NLP
A Python-based chatbot that uses Natural Language Processing (NLP) to understand and respond to user queries. The bot is trained on an intents.json dataset and uses a neural network (TensorFlow & TFLearn) for intent classification.
 ✨ Features
       Natural Language Understanding (NLU) using nltk
       Neural Network Model for intent classification
       Customizable responses via intents.json
       Interactive chat interface in the terminal
 🛠️ Tech Stack
       Python (Primary Language)
       NLTK (Natural Language Toolkit)
       TensorFlow (Deep Learning Framework)
       TFLearn (High-level TensorFlow API)
       NumPy (Numerical Computing)       
🚀 Installation & Setup
      1. Install Dependencies
             pip install nltk tensorflow tflearn numpy streamlit
             (Includes streamlit, nltk, tensorflow, numpy)
      2. Download NLTK Data
            import nltk
            nltk.download('punkt')
      3. Train the Model
            python train.py
            (Generates model.tflearn and words.pkl.)
      4. Run the Chatbot
            streamlit run app.py
            ➡️ Opens automatically at http://localhost:8501
 Example Output:
      User: Hi  
      Bot: Hello! How can I help you?





