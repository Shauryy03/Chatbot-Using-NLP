🤖 Chatbot Using NLP
A Python-based chatbot that uses Natural Language Processing (NLP) to understand and respond to user queries. The bot is trained on an intents.json dataset and uses a neural network (TensorFlow & TFLearn) for intent classification.

I. Project Directory Structure (Implicit): Based on the code, we can infer a basic project structure: your_project_directory/ ├── nltk_data/ # Directory where NLTK data (like punkt tokenizer) is downloaded ├── intents.json # File containing the chatbot's training data (intents, patterns, responses) ├── chat_log.csv # File to store the conversation history └── your_script_name.py # The Python script you provided

II. Implementation Details: A. Natural Language Processing (NLP) Component:

Libraries:
nltk: The Natural Language Toolkit is used here, specifically for downloading the punkt tokenizer, which is essential for breaking down text into words. sklearn.feature_extraction.text.TfidfVectorizer: This class from scikit-learn is used for converting text patterns into numerical vectors based on the Term Frequency-Inverse Document Frequency (TF-IDF) method. This representation helps the machine learning model understand the importance of words in the context of the entire set of patterns. The ngram_range=(1, 4) parameter indicates that the vectorizer will consider single words up to sequences of four consecutive words (n-grams) as features.
Tasks:
Intent Recognition: The primary NLP task here is to classify the user's input into a predefined intent. This is achieved by:
Vectorization: Transforming the user's input text into a TF-IDF vector using the vectorizer that was trained on the training patterns.
Classification: Using the trained LogisticRegression model (clf) to predict the intent (tag) associated with the input vector.
Model Training:
Data Loading: The intents.json file is loaded. This file likely contains a structure where each intent has a "tag," a list of "patterns" (user input examples), and a list of "responses."
Data Preprocessing: The code iterates through the intents data to create two lists: tags (the intent labels) and patterns (the corresponding user input examples).
Feature Extraction: The TfidfVectorizer is fitted to the patterns. This learns the vocabulary and IDF weights from the training data. The patterns are then transformed into a TF-IDF matrix x.
Model Training: A LogisticRegression classifier is initialized and trained using the TF-IDF features (x) and the intent labels (y). random_state=0 ensures reproducibility, and max_iter=10000 sets a limit on the number of iterations for the optimization algorithm to converge.
Model Storage (Implicit): While the code doesn't explicitly save the trained vectorizer and clf models to disk, in a more robust application, you would typically save these using libraries like pickle so you don't have to retrain them every time the application runs.
Implementation:
The chatbot(input_text) function takes user input, transforms it using the already fitted vectorizer, predicts the intent using the already trained clf model, and then randomly selects a response associated with that intent from the intents data. B. Machine Learning (ML) Component:
Libraries:
sklearn.linear_model.LogisticRegression: This is the core ML algorithm used for intent classification. Logistic Regression is a linear model that can be used for binary or multi-class classification tasks. It estimates the probability of an instance belonging to a particular class.
Tasks:
Intent Classification: The trained LogisticRegression model's primary task is to classify the vectorized user input into one of the predefined intent categories (tags).
Model Training: As described in the NLP section, the LogisticRegression model is trained on the TF-IDF representations of the user input patterns and their corresponding intent tags.
Implementation: The chatbot function utilizes the trained LogisticRegression model (clf.predict()) to determine the most likely intent for a given user input. C. Database Integration:
Library: csv: The csv module is used for working with CSV (Comma Separated Values) files. In this script, it's used to create and append to chat_log.csv, which acts as a simple form of storing conversation history.
Database System (Implicit): The chat_log.csv file serves as a rudimentary flat-file database to store the conversation history.
Connection Details (Implicit): The script directly opens and writes to the chat_log.csv file. There are no explicit connection parameters like host, user, or password, as it's a local file.
Data Schema (Implicit): The chat_log.csv file has a simple structure with three columns: "User Input," "Chatbot Response," and "Timestamp." The header row is written when the file is first created.
Operations:
Writing: When the user provides input and the chatbot generates a response, the user's input, the chatbot's response, and the current timestamp are appended as a new row to chat_log.csv.
Reading: In the "Conversation History" section of the Streamlit app, the script opens and reads the chat_log.csv file, displaying each row (conversation turn) to the user. D. API Integration:
None Explicitly Used: The provided code does not explicitly integrate with any external APIs. It relies solely on the intents.json file for its knowledge and responses. E. External Libraries:
os: Used for interacting with the operating system, such as checking if a file exists (os.path.exists()) and getting the absolute path of a file (os.path.abspath()).
json: Used for working with JSON data, specifically for loading the intents.json file.
datetime: Used for getting the current date and time to timestamp the conversation history.
csv: Used for reading and writing CSV files to store the conversation log.
nltk: As mentioned before, for NLP tasks, specifically downloading the punkt tokenizer.
ssl: Used here to bypass SSL certificate verification (ssl._create_default_https_context = ssl._create_unverified_context). This is generally not recommended for production environments as it can pose security risks. It's often used in development or when facing issues with SSL certificates.
streamlit: The core library for building the interactive web interface for the chatbot.
random: Used to randomly select a response from the list of responses associated with a matched intent.
sklearn.feature_extraction.text.TfidfVectorizer: For converting text to numerical vectors.
sklearn.linear_model.LogisticRegression: The machine learning model used for intent classification. F. Main Chatbot Logic:
chatbot(input_text) function: This function encapsulates the core logic of processing user input and generating a response:
Takes input_text as an argument.
Transforms the input text into a TF-IDF vector using the fitted vectorizer.
Uses the trained clf model to predict the intent (tag).
Iterates through the intents data to find the intent matching the predicted tag.
Randomly selects a response from the responses list of that intent.
Returns the selected response. G. Streamlit Interface (main() function):
st.title("Intents of Chatbot using NLP"): Sets the title of the web application.
Sidebar Menu:
st.sidebar.selectbox("Menu", ["Home", "Conversation History", "About"]): Creates a dropdown menu in the sidebar with three options.
"Home" Section:
Displays a welcome message.
Checks if chat_log.csv exists and creates it with a header if it doesn't.
Uses st.text_input("You:", key=f"user_input_{counter}") to create a text input box for the user to type their message. The key parameter is important for Streamlit to manage the state of the input across reruns. The counter variable ensures that each input box has a unique key.
If the user enters text:
The chatbot() function is called to get a response.
st.text_area("Chatbot:", value=response, ...) displays the chatbot's response in a text area.
The current timestamp is obtained.
The user's input, chatbot's response, and timestamp are written to chat_log.csv.
If the chatbot's response is "goodbye" or "bye" (case-insensitive), a farewell message is displayed, and st.stop() halts further execution.
"Conversation History" Section:
Displays a header.
Uses st.beta_expander("Click to see Conversation History") (note: beta_expander might be deprecated; st.expander is the current way) to create a collapsible section.
Opens and reads chat_log.csv, skipping the header row.
For each row in the CSV, it displays the user input, chatbot response, and timestamp using st.text() and adds a separator using st.markdown("---").
"About" Section:
Provides information about the project, including its goal, overview, the dataset used (implicitly the intents.json structure), the Streamlit chatbot interface, and a conclusion with potential future extensions.
if name == 'main': main(): This ensures that the main() function is executed when the script is run directly. H. Configuration:
intents.json: This file acts as the primary configuration for the chatbot's knowledge base, defining the intents, the patterns that trigger them, and the possible responses. The structure of this file is crucial for the chatbot's behavior.































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





