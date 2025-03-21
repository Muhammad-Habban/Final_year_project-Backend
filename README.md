# Book Vision RAG Pipeline

Book Vision is a backend system designed to handle user interactions with PDF documents, enabling users to create chats, ask questions, and retrieve responses using a Retrieval-Augmented Generation (RAG) pipeline. The system uses MongoDB for storing users, chats, and messages, and SQLite3 for storing document chunks and embeddings. Faiss is used for efficient similarity search, and Sentence Transformers are used for generating embeddings.

---

## Project Structure

### Models
- **User**: Represents a user in the system.
- **Chat**: Represents a chat session associated with a user and a PDF document.
- **Message**: Represents a message in a chat, including user prompts and LLM responses.

### Repositories
- **user_repository**: Handles database operations for the `User` model.
- **chat_repository**: Handles database operations for the `Chat` model.
- **message_repository**: Handles database operations for the `Message` model.

### Services
- **user_service**: Contains business logic for user-related operations.
- **chat_service**: Contains business logic for chat-related operations.
- **message_service**: Contains business logic for message-related operations.

### Controllers
- **user_controller**: Exposes endpoints for user management.
  - `/login`: User login.
  - `/signup`: User registration.
  - `/me`: Retrieves information about the logged-in user.
- **chat_controller**: Exposes endpoints for chat management.
  - `/create_chat`: Creates a chat by processing a PDF and generating chunks.
  - `/all-chat`: Retrieves all chats in the database.
  - `/user-chats`: Retrieves chats for a specific user.
- **message_controller**: Exposes endpoints for message management.
  - `/messages`: Retrieves all messages.
  - `/getresponse`: Sends a user prompt to the LLM and retrieves a response.
  - `/chat-messages`: Retrieves messages for a specific chat.

---

## Technologies Used
- **Database**:
  - MongoDB: Stores users, chats, and messages.
  - SQLite3: Stores document chunks and embeddings.
- **Embeddings**:
  - Sentence Transformers: Generates embeddings for document chunks.
- **Similarity Search**:
  - Faiss: Performs efficient cosine similarity search.
- **LLM Integration**:
  - A pre-trained language model (e.g., OpenAI GPT, Hugging Face models) for generating responses.

---

## Setup Instructions

### Prerequisites
1. Python 3.8 or higher.
2. MongoDB installed and running.
3. SQLite3 installed.
4. Faiss and Sentence Transformers installed.
