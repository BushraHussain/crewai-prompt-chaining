import time
import os
from dotenv import load_dotenv 
from litellm import completion
from crewai.flow.flow import Flow, start, listen

load_dotenv()

class BlogFlow(Flow):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Please set it in your .env file.")
        
        os.environ["GEMINI_API_KEY"] = self.api_key  # Set the API key for the runtime
        self.model = "gemini/gemini-1.5-pro-latest"

    def safe_completion(self, messages, max_retries=3, delay=5):
        """Handles API rate limits by retrying."""
        for attempt in range(max_retries):
            try:
                response = completion(model=self.model, messages=messages)
                return response
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)  # Wait before retrying
                else:
                    raise e  # Raise other errors immediately
        print("Max retries reached. Skipping this request.")
        return {"choices": [{"message": {"content": "No response due to rate limits"}}]}

    @start()
    def generate_topic(self):
        response = self.safe_completion([
            {"role": "user", "content": "Generate a trending 2025 topic for my blog. Topic length should be 5 words."}
        ])
        topic = response.get("choices", [{}])[0].get("message", {}).get("content", "No topic generated")
        print(f"Generated topic: {topic}")
        return topic

    @listen(generate_topic)
    def generate_outline(self, topic):
        response = self.safe_completion([
            {"role": "user", "content": f"Generate an outline for a blog post on {topic}. Outline length should be 50 words."}
        ])
        outline = response.get("choices", [{}])[0].get("message", {}).get("content", "No outline generated")
        print(f"Generated outline: {outline}")

        if not self.validate_outline(outline):
            print("Outline does not meet the criteria. Regenerating...")
            return self.generate_outline(topic)  # Retry outline generation

        return outline

    def validate_outline(self, outline):
        """Gate function to validate the outline before proceeding."""
        print("Validating outline...")
        return len(outline.split()) >= 30  # Outline must have at least 30 words

    @listen(generate_outline)
    def generate_blog_content(self, outline):
        response = self.safe_completion([
            {"role": "user", "content": f"Generate a blog post of 200 words based on the following outline: {outline}"}
        ])
        blog_content = response.get("choices", [{}])[0].get("message", {}).get("content", "No content generated")
        print(f"Generated blog content: {blog_content}")
        return blog_content

def main():
    flow = BlogFlow()
    flow.kickoff()
