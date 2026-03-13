"""
Simple pharmaceutical supply chain agent
Uses Llama3 + FAISS RAG
"""

import json
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class PharmaSupplyChainAgent:

    def __init__(self):
        print("🤖 Initializing Pharma Supply Chain Agent...\n")

        # Load company profile
        with open("data/company/company_profile.json") as f:
            self.company = json.load(f)

        # Load inventory
        with open("data/company/current_inventory.json") as f:
            self.inventory = json.load(f)

        print("📊 Loading FAISS supplier database...")

        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load FAISS database
        vectorstore = FAISS.load_local(
            "database/faiss_suppliers",
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        print("🧠 Loading Llama3 model via Ollama...")

        # Load LLM
        llm = Ollama(
            model="llama3",
            temperature=0.1
        )

        # Prompt template
        template = """
You are a pharmaceutical supply chain expert.

Use the supplier database context below to answer the question.

Context:
{context}

Question:
{question}

Give a helpful answer about pharmaceutical suppliers.
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )

        print("✅ Agent ready!\n")

    def ask(self, question):
        result = self.qa_chain.run(question)
        return result


# Run agent interactively
if __name__ == "__main__":

    print("🚀 Starting Pharma Supply Chain Agent\n")

    agent = PharmaSupplyChainAgent()

    while True:
        query = input("\nAsk a supplier question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        print("\n🔎 Searching supplier database...\n")

        response = agent.ask(query)

        print("💡 Answer:\n")
        print(response)