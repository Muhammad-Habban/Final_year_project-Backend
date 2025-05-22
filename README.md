# Book Vision Backend

This is a FastAPI-based backend application that provides AI-powered chat functionality with document processing capabilities. The application supports multiple AI models, PDF document processing, and interactive chat features.

## Features

- User authentication and authorization
- PDF document processing and chunking
- Multiple AI model support (OpenAI GPT-4, Google Gemini, Deepseek)
- Hybrid search functionality combining FAISS and BM25
- Chat history management
- YouTube video search integration
- LaTeX beamer slide generation
- Quiz generation from text

## Prerequisites

- Python 3.8+
- MongoDB
- Required Python packages (see requirements.txt)
- API keys for:
  - OpenAI
  - Google (for Gemini and Custom Search)
  - YouTube Data API

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Muhammad-Habban/Final_year_project-Backend
cd https://github.com/Muhammad-Habban/Final_year_project-Backend
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
YOUTUBE_API_KEY=your_youtube_api_key
LLAMA_MODEL_PATH=path_to_llama_model
```

## Running the Application

1. Start MongoDB service

2. Run the FastAPI application:

```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`

## API Endpoints

### User Management

#### POST /signup

- Creates a new user account
- Request body: `{ "email": string, "password": string }`
- Returns: User details with ID

#### POST /login

- Authenticates user and returns access tokens
- Request body: `{ "username": string, "password": string }`
- Returns: Access and refresh tokens

#### GET /me

- Returns details of currently logged-in user
- Requires authentication
- Returns: User details

### Chat Management

#### POST /create-chat

- Uploads a PDF and creates a new chat
- Form data: `file` (PDF), `user_id`
- Returns: Chat ID and success message

#### GET /all-chats

- Retrieves all chats
- Returns: List of all chats

#### GET /user-chats/{user_id}

- Retrieves chats for a specific user
- Returns: List of user's chats

#### PUT /edit-chat/{chat_id}

- Updates chat details
- Query params: `title` (optional)
- Returns: Success message

#### DELETE /delete-chat/{chat_id}

- Deletes a specific chat
- Returns: Success message

### Message Management

#### GET /messages/{chat_id}

- Retrieves all messages for a specific chat
- Returns: List of messages

### AI Model Endpoints

#### POST /open_ai_response

- Generates response using OpenAI GPT-4
- Query params: `chat_id`, `user_prompt`
- Returns: Generated message

#### POST /open_ai_response_with_context

- Generates response using GPT-4 with document context
- Query params: `chat_id`, `user_prompt`
- Returns: Generated message with context

#### POST /deepseek_response

- Generates response using Deepseek model
- Query params: `chat_id`, `user_prompt`
- Returns: Generated message

#### POST /enhanced_response_deepseek

- Generates enhanced response using Deepseek with context
- Query params: `chat_id`, `user_prompt`
- Returns: Generated message with context

#### POST /deepseek_q_response

- Generates response using quantized Deepseek model
- Query params: `chat_id`, `user_prompt`
- Returns: Generated message

#### POST /gemini_flash_response

- Generates response using Google Gemini Flash
- Query params: `chat_id`, `user_prompt`
- Returns: Generated message

#### POST /enhanced_gemini_response

- Generates enhanced response using Gemini with context
- Query params: `chat_id`, `user_prompt`
- Returns: Generated message with context

### Additional Features

#### POST /generate-quiz

- Generates quiz questions from provided text
- Query params: `user_prompt`
- Returns: JSON with quiz questions and answers

#### POST /search-with-images

- Searches web and returns results with images
- Query params: `query`
- Returns: Web results with associated images

#### POST /generate-beamer-slide

- Generates LaTeX beamer slides from message content
- Query params: `message_id`
- Returns: PDF path

#### GET /youtube-search

- Searches YouTube videos
- Query params: `q`, `max_results` (optional)
- Returns: List of video results

## Project Structure

```
├── controllers/
│   ├── chat_controller.py
│   ├── message_controller.py
│   └── user_controller.py
├── models/
│   ├── chat.py
│   ├── message.py
│   └── user.py
├── repositories/
│   ├── chat_repository.py
│   ├── message_repository.py
│   └── user_repository.py
├── services/
│   ├── chat_service.py
│   ├── message_service.py
│   └── user_service.py
├── database.py
├── main.py
└── requirements.txt
```

## Security

- Password hashing using bcrypt
- JWT token-based authentication
- Secure password storage
- API key management through environment variables

## Error Handling

The application includes comprehensive error handling for:

- Invalid file types
- Authentication failures
- Database errors
- API rate limiting
- Invalid requests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
