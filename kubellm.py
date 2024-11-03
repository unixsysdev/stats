# Standard library imports
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# Third-party imports
import requests

class KubectlLLMAssistant:
    def __init__(self, debug=False):
        # Updated endpoint with the correct API path
        self.endpoint = "https://marcelbp--example-vllm-openai-compatible-serve.modal.run/v1/chat/completions"
        self.model = "Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
        self.token = ""
        
        # Setup logging
        self.setup_logging(debug)
        self.logger = logging.getLogger('KubectlLLM')

    def setup_logging(self, debug=True):
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Create a timestamp for the log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/kubellm_{timestamp}.log'

        # Setup logging format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

        # Setup root logger
        logger = logging.getLogger('KubectlLLM')
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log initial setup
        logger.info(f"Starting new KubectlLLM session. Log file: {log_file}")
        logger.info(f"Endpoint: {self.endpoint}")
        logger.info(f"Model: {self.model}")

    def get_cluster_state(self) -> str:
        """Get current cluster state."""
        self.logger.info("Fetching current cluster state...")
        try:
            result = subprocess.run(
                    # only a part of the cluster state is sent
                ["kubectl", "get", "all"], #, "-A", "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.debug(f"Cluster state output:\n{result.stdout}")
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get cluster state: {e}")
            self.logger.debug(f"Error output: {e.stderr}")
            return ""

    def create_prompt(self, user_request: str, cluster_state: str) -> str:
        """Create a structured prompt for the LLM."""
        self.logger.info("Creating prompt for LLM")
        prompt = f"""You are a Kubernetes expert assistant. Please help with the following request:

Existing cluster configuration in json:
"{cluster_state}"

The request coming from the user:
"{user_request}"

Please provide your response in the following JSON format:
{{
    "files_needed": [
        {{
            "filename": "string",
            "contents": "string"
        }}
    ],
    "steps": [
        {{
            "description": "string",
            "command": "string"
        }}
    ]
}}

Important guidelines:
1. If no files are needed, return an empty array for files_needed
2. Each step must include both a description and the exact kubectl command
3. Use proper Kubernetes best practices
4. Ensure commands are idempotent where possible
5. Include verification steps when appropriate
6. Do no include markup tags
7. Only answer with the json response nothing else, you are basically and API endpoint that speaks computer
8. Each step should only include one commmand, don't chain commands
9. In case of error, or you can answer reply with json that has error as the root element
"""
        self.logger.debug(f"Generated prompt:\n{prompt}")
        return prompt

    def query_llm(self, prompt: str) -> Dict:
        """Send request to LLM endpoint."""
        self.logger.info("Sending request to LLM endpoint...")
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000  # Added to ensure we get complete responses
        }
        
        self.logger.debug(f"Request data (excluding prompt content):\n{json.dumps({k:v for k,v in data.items() if k != 'messages'}, indent=2)}")
        self.logger.debug(f"Using endpoint: {self.endpoint}")
        
        try:
            self.logger.debug("Sending POST request to endpoint...")
            response = requests.post(self.endpoint, headers=headers, json=data)
            
            # Add more detailed error logging
            if response.status_code != 200:
                self.logger.error(f"LLM request failed with status code: {response.status_code}")
                self.logger.debug(f"Response content: {response.text}")
                self.logger.debug(f"Request headers: {headers}")
                self.logger.debug(f"Request endpoint: {self.endpoint}")
                response.raise_for_status()
            
            content = response.json()["choices"][0]["message"]["content"]
            self.logger.debug(f"LLM response:\n{content}")
            return content
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to query LLM: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.debug(f"Error response: {e.response.text}")
                self.logger.debug(f"Error status code: {e.response.status_code}")
            else:
                self.logger.debug("No response object available")
            sys.exit(1)


    def create_files(self, files: List[Dict]) -> None:
        """Create any necessary files."""
        if not files:
            self.logger.info("No files need to be created")
            return
        
        self.logger.info(f"Need to create {len(files)} file(s)")
        print("\nThe following files need to be created:")
        for file in files:
            print(f"\nFile: {file['filename']}")
            print("Contents:")
            print(file['contents'])
            self.logger.debug(f"File to create: {file['filename']}\nContents:\n{file['contents']}")
            
        if input("\nWould you like to create these files? (y/N): ").lower() != 'y':
            self.logger.info("User chose not to create files. Exiting...")
            sys.exit(0)
            
        for file in files:
            try:
                with open(file['filename'], 'w') as f:
                    f.write(file['contents'])
                self.logger.info(f"Successfully created file: {file['filename']}")
            except IOError as e:
                self.logger.error(f"Failed to create file {file['filename']}: {e}")
                sys.exit(1)

    def execute_steps(self, steps: List[Dict]) -> None:
        """Execute kubectl commands step by step."""
        self.logger.info(f"Starting execution of {len(steps)} step(s)")
        
        for i, step in enumerate(steps, 1):
            self.logger.info(f"Processing step {i}/{len(steps)}")
            self.logger.debug(f"Step details: {json.dumps(step, indent=2)}")
            
            print(f"\nStep {i}: {step['description']}")
            print(f"Command: {step['command']}")
            
            if input("\nExecute this step? (y/N): ").lower() != 'y':
                self.logger.info(f"User skipped step {i}")
                print("Skipping step...")
                continue
                
            try:
                self.logger.debug(f"Executing command: {step['command']}")
                result = subprocess.run(
                    step['command'].split(),
                    capture_output=True,
                    text=True,
                    check=True
                )
                print("\nOutput:")
                print(result.stdout)
                self.logger.debug(f"Command output:\n{result.stdout}")
                
                # Add a small delay between commands
                time.sleep(1)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Command execution failed: {e}")
                self.logger.debug(f"Error output: {e.stderr}")
                if input("\nContinue with remaining steps? (y/N): ").lower() != 'y':
                    self.logger.info("User chose to abort remaining steps")
                    sys.exit(1)

def clean_json_string(s: str) -> str:
    """Clean and validate JSON string before parsing."""
    # Remove any potential Unicode BOM and whitespace
    s = s.strip().lstrip('\ufeff')
    
    # Remove any trailing commas before closing braces/brackets
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    
    return s

def check_for_error_in_json(response_dict: dict) -> Optional[str]:
    """
    Recursively check for error messages in the JSON response.
    Returns the error message if found, None otherwise.
    """
    if isinstance(response_dict, dict):
        # Check for common error field names
        error_fields = ['error', 'errors', 'message', 'errorMessage', 'error_message']
        for field in error_fields:
            if field in response_dict:
                return response_dict[field]
        
        # Recursively check nested dictionaries
        for value in response_dict.values():
            error = check_for_error_in_json(value)
            if error:
                return error
                
    elif isinstance(response_dict, list):
        # Recursively check items in lists
        for item in response_dict:
            error = check_for_error_in_json(item)
            if error:
                return error
                
    return None

def main():
    # Setup argument parsing
    if len(sys.argv) < 2:
        print("Usage: python script.py <kubectl command>")
        sys.exit(1)
    
    # Get the user's request (everything after the script name)
    user_request = " ".join(sys.argv[1:])
    
    # Initialize assistant with debugging enabled
    assistant = KubectlLLMAssistant()
    logger = logging.getLogger('KubectlLLM')
    
    logger.info(f"Processing user request: {user_request}")
    
    try:
        # Get current cluster state
        cluster_state = assistant.get_cluster_state()
        
        # Create prompt and query LLM
        prompt = assistant.create_prompt(user_request, cluster_state)
        response = assistant.query_llm(prompt)
        
        # Clean and parse JSON response
        logger.debug("Attempting to parse LLM response as JSON...")
        try:
            cleaned_response = clean_json_string(response)
            logger.debug(f"Cleaned response:\n{cleaned_response}")
            
            parsed_response = json.loads(cleaned_response)
            
            # Check for errors in the JSON response
            if error_message := check_for_error_in_json(parsed_response):
                logger.error(f"LLM returned an error: {error_message}")
                print(f"\nError from LLM: {error_message}")
                sys.exit(1)
            
            # Validate required fields
            if not isinstance(parsed_response, dict):
                raise ValueError("Response must be a JSON object")
                
            if 'steps' not in parsed_response:
                raise ValueError("Response must contain 'steps' field")
                
            if not isinstance(parsed_response.get('files_needed', []), list):
                raise ValueError("'files_needed' must be a list if present")
                
            logger.info("Successfully parsed and validated JSON response")
            
            # Create any necessary files
            assistant.create_files(parsed_response.get('files_needed', []))
            
            # Execute steps
            assistant.execute_steps(parsed_response['steps'])
            
            logger.info("Command execution completed successfully")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.debug(f"Error location: line {e.lineno}, column {e.colno}")
            logger.debug(f"Error message: {e.msg}")
            logger.debug("Full response that failed to parse:")
            logger.debug(response)
            print("\nError: Invalid JSON response from LLM")
            print(f"JSON parsing error: {str(e)}")
            print("\nResponse received:")
            print(response)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
