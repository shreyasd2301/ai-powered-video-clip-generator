import json
import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

from services.logger_service import get_logger

load_dotenv()

# Initialize logger
logger = get_logger("llm_service")


class LLMService:
    def __init__(self, llm_type: str = "openai", model: str = "gpt-4o"):
        self.llm_type = llm_type
        self.model = model
        self.openai_key = os.getenv("OPENAI_API_KEY")
        logger.info(f"LLMService initialized with type={llm_type}, model={model}")
    
    def chat(self, message, functions=None):
        """Direct chat method - merged from LLM class"""
        message = [self._to_gpt_msg(message)]
        return self._call_openai(message, functions)


    def _to_gpt_msg(self, data):
        """Convert data to GPT message format"""
        context_msg = ""
        context_msg += str(data)
        return {"role": "system", "content": context_msg}

    def _call_openai(self, message, functions=None):
        """Call OpenAI API - merged from LLM class"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}",
        }
        data = {
            "model": self.model,
            "messages": message,
            "temperature": 0.1,
        }
        data["response_format"] = {"type": "json_object"}
        if functions:
            data.update(
                {
                    "functions": functions,
                    "function_call": "auto",
                }
            )

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response_json = response.json()
            
            if response.status_code != 200:
                error_msg = response_json.get("error", {}).get("message", "Unknown error")
                return {"error": f"OpenAI API Error ({response.status_code}): {error_msg}"}
            
            return response_json
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Failed to decode JSON response from OpenAI API"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    async def generate_response(self, message: str, functions: Optional[List] = None) -> Dict:
        """Generate response using LLM"""
        logger.debug(f"Generating LLM response for message: {message[:100]}...")
        try:
            response = self.chat(message=message, functions=functions)
            logger.debug("LLM response generated successfully")
            return response
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {str(e)}")
            raise Exception(f"Failed to generate LLM response: {str(e)}")
    
    async def rank_results(self, results: List[str], user_query: str) -> List[tuple]:
        """Rank results using LLM"""
        logger.debug(f"Ranking {len(results)} results for query: {user_query[:50]}...")
        try:
            ranked_results = []
            for text in results:
                score = self._ranking_prompt_llm(text, user_query)
                ranked_results.append((text, score))
            
            # Sort by score (descending)
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"Ranking completed, top score: {ranked_results[0][1] if ranked_results else 0}")
            return ranked_results
        
        except Exception as e:
            logger.error(f"Failed to rank results: {str(e)}")
            raise Exception(f"Failed to rank results: {str(e)}")
    
    def _ranking_prompt_llm(self, text: str, user_query: str) -> float:
        """Rank results using LLM - simplified version for llm_service"""
        try:
            ranking_prompt = f"""Given the text provided below and a specific User Prompt, evaluate the relevance of the text
            in relation to the user's prompt. Please assign a relevance score ranging from 0 to 10, where 0 indicates no relevance 
            and 10 signifies perfect alignment with the user's request.
            The score quality also increases when the text is a complete sentence, making it perfect for a video clip result

            text: {text}
            User Prompt: {user_query}

            Ensure the final output strictly adheres to the JSON format specified, without including additional text or explanations. 
            Use the following structure for your response:
            {{
            "score": <relevance score>
            }}
            """
            
            response = self.chat(message=ranking_prompt)
            output = response["choices"][0]["message"]["content"]
            res = json.loads(output)
            score = res.get('score', 0)
            return score
        except Exception as e:
            # Return 0 score if ranking fails
            return 0 