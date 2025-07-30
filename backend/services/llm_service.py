import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Import existing LLM functionality
import sys
sys.path.append('../..')
from llm_agent import LLM, LLMType, Models
from services.logger_service import get_logger

load_dotenv()

# Initialize logger
logger = get_logger("llm_service")

class LLMService:
    def __init__(self, llm_type: str = "openai", model: str = "gpt-4"):
        self.llm_type = llm_type
        self.model = model
        self.llm = LLM(llm_type=LLMType.OPENAI, model=Models.GPT4)
        logger.info(f"LLMService initialized with type={llm_type}, model={model}")
    
    async def generate_response(self, message: str, functions: Optional[List] = None) -> Dict:
        """Generate response using LLM"""
        logger.debug(f"Generating LLM response for message: {message[:100]}...")
        try:
            response = self.llm.chat(message=message, functions=functions)
            logger.debug("LLM response generated successfully")
            return response
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {str(e)}")
            raise Exception(f"Failed to generate LLM response: {str(e)}")
    
    async def rank_results(self, results: List[str], user_query: str) -> List[tuple]:
        """Rank results using LLM"""
        logger.debug(f"Ranking {len(results)} results for query: {user_query[:50]}...")
        try:
            from llm_agent import ranking_prompt_llm
            
            ranked_results = []
            for text in results:
                score = ranking_prompt_llm(text, user_query)
                ranked_results.append((text, score))
            
            # Sort by score (descending)
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"Ranking completed, top score: {ranked_results[0][1] if ranked_results else 0}")
            return ranked_results
        
        except Exception as e:
            logger.error(f"Failed to rank results: {str(e)}")
            raise Exception(f"Failed to rank results: {str(e)}")
    
    async def suggest_queries(self, video_content: str) -> List[str]:
        """Suggest queries based on video content"""
        try:
            prompt = f"""
            Based on the following video content, suggest 5 interesting queries that could be used to generate clips.
            Focus on different aspects like humor, education, highlights, emotions, etc.
            
            Video Content: {video_content[:1000]}...
            
            Return only the queries as a JSON list:
            ["query1", "query2", "query3", "query4", "query5"]
            """
            
            response = await self.generate_response(prompt)
            # Parse response to extract queries
            # This is a simplified implementation
            return [
                "find funny moments",
                "find educational content", 
                "find highlights",
                "find emotional moments",
                "find key insights"
            ]
        
        except Exception as e:
            raise Exception(f"Failed to suggest queries: {str(e)}")
    
    async def analyze_video_content(self, transcript: str) -> Dict:
        """Analyze video content and provide insights"""
        try:
            prompt = f"""
            Analyze the following video transcript and provide insights about:
            1. Main topics discussed
            2. Key themes
            3. Potential clip opportunities
            4. Content type (educational, entertainment, etc.)
            
            Transcript: {transcript[:2000]}...
            
            Return as JSON:
            {{
                "topics": ["topic1", "topic2"],
                "themes": ["theme1", "theme2"],
                "clip_opportunities": ["opportunity1", "opportunity2"],
                "content_type": "educational/entertainment/informational"
            }}
            """
            
            response = await self.generate_response(prompt)
            # Parse response to extract analysis
            # This is a simplified implementation
            return {
                "topics": ["Main topic 1", "Main topic 2"],
                "themes": ["Theme 1", "Theme 2"],
                "clip_opportunities": ["Funny moments", "Key insights"],
                "content_type": "educational"
            }
        
        except Exception as e:
            raise Exception(f"Failed to analyze video content: {str(e)}")
    
    async def optimize_query(self, original_query: str, video_context: str) -> str:
        """Optimize user query for better results"""
        try:
            prompt = f"""
            Optimize the following user query to get better video clip results.
            Make it more specific and targeted while maintaining the original intent.
            
            Original Query: {original_query}
            Video Context: {video_context[:500]}...
            
            Return only the optimized query as a string.
            """
            
            response = await self.generate_response(prompt)
            # Parse response to extract optimized query
            # This is a simplified implementation
            return original_query  # Return original for now
        
        except Exception as e:
            raise Exception(f"Failed to optimize query: {str(e)}")
    
    def get_available_models(self) -> Dict:
        """Get available LLM models"""
        return {
            "openai": {
                "gpt-3.5-turbo": "Fast and cost-effective",
                "gpt-4": "High quality, recommended",
                "gpt-4o": "Latest model, best performance"
            },
            "claude": {
                "claude-instant-1.1": "Fast Claude model",
                "claude-2": "High quality Claude model"
            },
            "gemini": {
                "gemini-1.5-flash": "Fast Gemini model",
                "gemini-1.5-pro": "High quality Gemini model"
            }
        }
    
    def switch_model(self, llm_type: str, model: str):
        """Switch to different LLM model"""
        try:
            if llm_type == "openai":
                self.llm = LLM(llm_type=LLMType.OPENAI, model=model)
            elif llm_type == "claude":
                self.llm = LLM(llm_type=LLMType.CLAUDE, model=model)
            elif llm_type == "gemini":
                self.llm = LLM(llm_type=LLMType.GEMINI, model=model)
            else:
                raise Exception(f"Unsupported LLM type: {llm_type}")
            
            self.llm_type = llm_type
            self.model = model
            
        except Exception as e:
            raise Exception(f"Failed to switch model: {str(e)}") 