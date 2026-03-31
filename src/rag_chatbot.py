"""
RAG + Agentic AI Chatbot
========================
A customer support chatbot powered by Retrieval-Augmented Generation (RAG)
and a LangChain agent for multi-step reasoning and personalized responses.

Architecture:
  Customer query
      → Sentiment check (guard)
      → RAG retrieval (FAISS + product knowledge base)
      → LangChain Agent (tool use: lookup order, fetch profile, recommend)
      → Personalized GPT-4o response
      → Response stored → feedback loop
"""

import os
from typing import Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── LLM & Embeddings ───────────────────────────────────────────────────────
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

def get_embeddings():
    return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


# ── Knowledge Base (RAG) ───────────────────────────────────────────────────
KNOWLEDGE_DOCS = [
    "Our return policy allows returns within 30 days with receipt.",
    "Standard delivery takes 3-5 business days. Express is 1-2 days.",
    "Loyalty members get 10% off every purchase and free shipping on orders over 500k VND.",
    "To track your order, use the order ID in 'My Orders' section of the app.",
    "Customer support is available Mon-Sat, 8AM-10PM via chat or hotline 1800-xxx.",
    "We offer installment payments (0% interest) for purchases above 2 million VND.",
]

def build_knowledge_base(docs: list[str] = KNOWLEDGE_DOCS):
    """Build FAISS vector store from knowledge base documents."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    documents = [Document(page_content=d) for d in docs]
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ── Agent Tools ────────────────────────────────────────────────────────────
@tool
def lookup_order_status(order_id: str) -> str:
    """Look up the current status of a customer order by order ID."""
    # In production: query PostgreSQL orders table
    mock_orders = {
        "ORD-001": "Shipped — Expected delivery: Apr 3, 2026",
        "ORD-002": "Processing — Will ship within 24 hours",
        "ORD-003": "Delivered on Mar 28, 2026",
    }
    return mock_orders.get(order_id, f"Order {order_id} not found. Please check your order ID.")


@tool
def get_customer_profile(customer_id: str) -> str:
    """Retrieve customer profile, tier, and purchase history summary."""
    # In production: SELECT * FROM customers WHERE id = customer_id
    mock_profiles = {
        "CUST-101": "Name: Linh. Tier: Gold. Last purchase: 5 days ago. Avg order: 850k VND. Preferred category: Skincare.",
        "CUST-202": "Name: Minh. Tier: Silver. Last purchase: 32 days ago. At-risk: churn signal detected.",
    }
    return mock_profiles.get(customer_id, "Customer profile not found.")


@tool
def get_next_best_action(customer_id: str) -> str:
    """Get the top personalized recommendation action for the customer."""
    # In production: call recommendation_engine.get_recommendations()
    mock_actions = {
        "CUST-101": "Recommend: Vitamin C Serum (matches past Skincare purchases). Offer: 15% loyalty discount.",
        "CUST-202": "Trigger win-back campaign: Send 20% re-engagement voucher. Use positive sentiment tone.",
    }
    return mock_actions.get(customer_id, "No personalized action available. Offer general promotion.")


# ── Agentic Chatbot ────────────────────────────────────────────────────────
class CustomerNurturingChatbot:
    """
    RAG + Agentic chatbot for personalized customer support.

    Combines:
    - RAG for factual Q&A (policies, shipping, etc.)
    - Agent tools for real-time data (orders, profiles, recommendations)
    - Sentiment-aware tone adjustment
    """

    def __init__(self, knowledge_docs: list[str] = KNOWLEDGE_DOCS):
        self.llm        = get_llm()
        self.vectorstore = build_knowledge_base(knowledge_docs)
        self.retriever  = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.tools      = [lookup_order_status, get_customer_profile, get_next_best_action]
        self.agent      = self._build_agent()

    def _build_agent(self) -> AgentExecutor:
        system_prompt = """You are Aria, a friendly and empathetic AI customer service assistant 
for a Vietnamese e-commerce brand. Your goal is to resolve customer issues quickly 
and proactively offer personalized recommendations.

Guidelines:
- Always be warm, concise, and helpful.
- If sentiment is Negative, acknowledge the frustration first before solving.
- Use tools to get real data — never make up order statuses or policies.
- After resolving the main issue, suggest one relevant next action.
- Respond in the same language as the customer (Vietnamese or English).

You have access to: Company knowledge base, Order lookup, Customer profiles, Personalized recommendations.
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=4)

    def chat(
        self,
        user_message: str,
        customer_id: Optional[str] = None,
        sentiment: Optional[str] = None,
        chat_history: list = None,
    ) -> dict:
        """
        Process a customer message and return a personalized AI response.

        Parameters
        ----------
        user_message : str
        customer_id  : str | None — if provided, agent can personalize
        sentiment    : str | None — pre-classified sentiment to adjust tone
        chat_history : list — LangChain message history for context

        Returns
        -------
        dict: {response, sources, tools_used}
        """
        # Step 1: RAG context enrichment
        rag_docs = self.retriever.get_relevant_documents(user_message)
        rag_context = "\n".join([d.page_content for d in rag_docs])

        # Step 2: Build enriched input
        enriched_input = user_message
        if customer_id:
            enriched_input += f"\n[Customer ID: {customer_id}]"
        if sentiment:
            enriched_input += f"\n[Detected Sentiment: {sentiment}]"
        if rag_context:
            enriched_input += f"\n[Knowledge Base Context: {rag_context}]"

        # Step 3: Agent execution
        response = self.agent.invoke({
            "input": enriched_input,
            "chat_history": chat_history or [],
        })

        return {
            "response":   response["output"],
            "rag_sources": [d.page_content for d in rag_docs],
            "customer_id": customer_id,
            "sentiment":   sentiment,
        }


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    bot = CustomerNurturingChatbot()

    # Simulated chat session
    response = bot.chat(
        user_message="I ordered 3 days ago and still haven't received it.",
        customer_id="CUST-101",
        sentiment="Negative",
    )
    print("\n🤖 Aria:", response["response"])
    print("\n📚 RAG Sources:", response["rag_sources"])
