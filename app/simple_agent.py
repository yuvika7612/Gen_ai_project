"""
Simple pharmaceutical supply chain agent
Uses Llama3 + FAISS RAG + local GGUF model
"""

import json
from llama_cpp import Llama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


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

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.load_local(
            "database/faiss_suppliers",
            embeddings,
            allow_dangerous_deserialization=True
        )

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        print("🧠 Loading GGUF model...")

        # ✅ Point this to wherever you saved the .gguf file
        self.llm = Llama(
            model_path="models/llama-3-8b.Q4_K_M.gguf",  # <-- change this path
            n_ctx=4096,          # context window
            n_gpu_layers=0,     # -1 = all layers on GPU, 0 = CPU only
            chat_format="chatml" # Unsloth models typically use chatml
        )

        self.prompt_template = """You are a pharmaceutical supply chain expert.

Use the supplier database context below to answer the question.

Context:
{context}

Question:
{question}

Give a helpful answer about pharmaceutical suppliers."""

        print("✅ Agent ready!\n")

    def ask(self, question):
        # Retrieve relevant docs from FAISS
        docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build the prompt
        prompt = self.prompt_template.format(context=context, question=question)

        # Run inference using chat completion (works best with finetuned/instruct models)
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a pharmaceutical supply chain expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.1,
        )

        return response["choices"][0]["message"]["content"]


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
