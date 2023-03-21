from transformers import pipeline

# Load GPT-Neo model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B', device=0)

# Define function to generate response
def generate_response(prompt):
    response = generator(prompt, max_length=256, do_sample=True, temperature=0.7)
    return response[0]['generated_text']

# Define function to preprocess user input
def preprocess_input(input_text):
    # Remove special characters and convert to lowercase
    processed_input = re.sub('[^A-Za-z0-9 ]+', '', input_text).lower()
    return processed_input

# Define sample data
sample_data = {
    "greetings": ["Hello", "Hi", "Hey", "Howdy"],
    "goodbyes": ["Bye", "Goodbye", "See you later", "Take care"],
    "questions": ["What's your name?", "How are you?", "What do you do?", "Where are you from?"],
    "responses": ["My name is Chatbot.", "I'm doing well, thank you.", "I'm here to chat with you.", "I'm from the internet."]
}

# Define function to handle user input
def handle_input(input_text):
    # Preprocess user input
    input_text = preprocess_input(input_text)

    # Check if input matches any predefined questions
    for question, response in zip(sample_data["questions"], sample_data["responses"]):
        if question.lower() in input_text:
            return response

    # Check if input matches any predefined greetings
    for greeting in sample_data["greetings"]:
        if greeting.lower() in input_text:
            return "Hi there!"

    # Check if input matches any predefined goodbyes
    for goodbye in sample_data["goodbyes"]:
        if goodbye.lower() in input_text:
            return "Goodbye!"

    # If input doesn't match any predefined patterns, generate response using GPT-Neo
    return generate_response(input_text)

# Main loop to handle user input and generate responses
while True:
    input_text = input("You: ")
    response = handle_input(input_text)
    print("Chatbot:", response)
