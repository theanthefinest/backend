from langchain_router import Router
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_router():
    router = Router()

    test_queries = [
        "What are the best temples to visit in Siem Reap?",
        "Tell me about Angkor Wat history and architecture",
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Best time to visit Cambodia for good weather",
        "What are some good restaurants in Phnom Penh?",
        "How does photosynthesis work in plants?",
        "Tell me about Cambodian culture and traditions",
        "Where should I stay in Battambang?",
        "What is the theory of relativity?",
        "Recommend some traditional Cambodian dishes to try",
        "How do I apply for a visa to visit Cambodia?",
        "What are the main programming languages for web development?"
    ]
    
    print("🧪 Testing Smart Router LLM\n")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 🧑‍💻 Query: {query}")
        try:
            response = smart_llm(query)
            print(f"   🤖 Response: {response[:200]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("-" * 80)

def create_conversation_chain():

    router = Router()
    memory = ConversationBufferMemory()
    
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template="""Previous conversation:
{history}

Human: {input}
Assistant:"""
    )
    
    conversation_chain = LLMChain(
        llm=smart_llm,
        prompt=prompt_template,
        memory=memory,
        verbose=True
    )
    
    return conversation_chain

def test_conversation():
    
    conversation = create_conversation_chain()
    
    conversation_messages = [
        "Hello, I'm planning a trip to Cambodia next month.",
        "What are the must-see temples in Angkor?",
        "Also, what's the best Cambodian food I should try?",
        "By the way, what's the weather like in December?",
        "Can you also explain how blockchain technology works?"  # Should go to Gemini
    ]
    
    print("\n💬 Testing Conversation with Memory\n")
    print("=" * 80)
    
    for message in conversation_messages:
        print(f"\n🧑‍💻 User: {message}")
        try:
            response = conversation.run(input=message)
            print(f"🤖 Assistant: {response[:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 80)

if __name__ == "__main__":
    print("🚀 Starting Smart Router LLM Tests")
    print("⚠️  Make sure your FastAPI server is running on http://localhost:8000")
    print("=" * 80)
    
    test_router()

    test_conversation()
    
    print("\n✅ All tests completed!")