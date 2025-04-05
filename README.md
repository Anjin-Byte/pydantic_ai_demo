## Setting Up Your OpenAI API Key

For security, the OpenAI API key is stored in a `.env` file rather than in your source code. Follow these steps to set it up:

1. **Create a `.env` file**  
   In the root directory of the project, create a file named `.env`.

2. **Add Your OpenAI API Key**  
   Open the `.env` file in your text editor and add the following line:
   ```dotenv
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Load the Environment Variables in Your Code**  
   In your Python code, load the variables from `.env` by adding:
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()  # Load variables from .env into the environment
   openai_api_key = os.getenv("OPENAI_API_KEY")
   ```
