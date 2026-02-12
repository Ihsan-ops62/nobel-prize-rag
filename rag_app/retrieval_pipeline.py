from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import config
from langchain_groq import ChatGroq

class RetrievalPipeline:
    """
    Retrieval-based pipeline for answering Nobel Prize related questions.
    Handles greetings, ambiguity, invalid categories, and off-topic queries.
    """

    def __init__(self):
        # Initialize embedding model
        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )

        # Load existing Chroma vector store
        self.vectorstore = Chroma(
            persist_directory=str(config.CHROMA_PERSIST_DIRECTORY),
            embedding_function=self.embeddings,
            collection_name=config.CHROMA_COLLECTION_NAME
        )

        # Initialize LLM (low temperature for factual accuracy)
        # self.llm = ChatOlaama(
        #     model=config.OLLAMA_LLM_MODEL,
        #     base_url=config.OLLAMA_BASE_URL,
        #     temperature=0.0,
        #     top_p=0.9,
        #     num_ctx=4096
        # )
        self.llm = ChatGroq(
            model=config.GROQ_MODEL,
            temperature=0.7
        )
        # Prompt enforcing strict context usage
        prompt_template = """
You are a Nobel Prize information assistant.

Use only the provided context to answer the question.
Do not invent or assume information.
If the answer is not clearly present in the context, respond with:
The aswer should be accurate and to the point. Avoid unnecessary elaboration.
Do not include information that is not directly relevant to the question.
"Sorry, I don’t have information about that."
Don't mention the context or say "based on the provided information". Just provide the answer.
Format the answer clearly using headings and bullet points.

Context:
{context}

Question:
{question}

Answer:
"""

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": config.RETRIEVAL_K}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def preprocess_query(self, query: str) -> str:
        """
        Classify user query into categories:
        greeting, identity, invalid category, ambiguous, nobel-related, off-topic
        """
        query_lower = query.strip().lower()

        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        identity_questions = ["who are you", "what are you", "introduce yourself"]

        valid_categories = [
            "physics",
            "chemistry",
            "medicine",
            "literature",
            "peace",
            "economic"
        ]

        # Greeting detection (exact match)
        if query_lower in greetings:
            return "greeting"

        # Identity questions
        if any(q in query_lower for q in identity_questions):
            return "identity"

        # Detect mathematics (no Nobel Prize exists)
        if "math" in query_lower:
            return "invalid_category"

        # Ambiguous first winner question without category
        if "first winner" in query_lower:
            if not any(cat in query_lower for cat in valid_categories):
                return "ambiguous"

        # Nobel-related keywords
        nobel_keywords = ["nobel", "prize", "winner", "laureate", "year"]

        if any(word in query_lower for word in nobel_keywords):
            return "nobel"

        return "off_topic"

    def format_answer(self, answer: str) -> str:
        """
        Clean answer formatting.
        """
        return answer.strip()

    def ask(self, query: str) -> str:
        """
        Main method for answering user queries.
        """
        try:
            query_type = self.preprocess_query(query)

            if query_type == "greeting":
                return "Hello! I’m here to help you with Nobel Prize information."

            if query_type == "identity":
                return "I am an AI assistant specialized in Nobel Prize information."

            if query_type == "invalid_category":
                return (
                    "There is no Nobel Prize in Mathematics.\n\n"
                    "The Nobel Prizes are awarded in:\n"
                    "• Physics\n"
                    "• Chemistry\n"
                    "• Physiology or Medicine\n"
                    "• Literature\n"
                    "• Peace\n"
                    "• Economic Sciences"
                )

            if query_type == "ambiguous":
                return (
                    "Please specify the category for the first winner.\n\n"
                    "Available categories:\n"
                    "• Physics\n"
                    "• Chemistry\n"
                    "• Physiology or Medicine\n"
                    "• Literature\n"
                    "• Peace\n"
                    "• Economic Sciences"
                )

            if query_type == "off_topic":
                return "Sorry, I only answer questions related to Nobel Prizes."

            # Nobel-related query → use retrieval chain
            result = self.qa_chain.invoke({"query": query})
            answer = result.get("result", "").strip()

            # Basic hallucination safeguard
            if (
                not answer
                or "not included in provided context" in answer.lower()
                or "not applicable" in answer.lower()
                or len(answer) < 10
            ):
                return "Sorry, I don’t have information about that."

            return self.format_answer(answer)

        except Exception as e:
            print(f"Error in ask(): {e}")
            return "An internal error occurred."

    def ask_with_sources(self, query: str) -> dict:
        """
        Returns answer along with retrieved source documents.
        """
        try:
            query_type = self.preprocess_query(query)

            if query_type != "nobel":
                return {
                    "query": query,
                    "answer": self.ask(query),
                    "source_documents": []
                }

            result = self.qa_chain.invoke({"query": query})
            answer = result.get("result", "").strip()

            if not answer or len(answer) < 10:
                answer = "Sorry, I don’t have information about that."

            return {
                "query": query,
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }

        except Exception as e:
            print(f"Error in ask_with_sources(): {e}")
            return {
                "query": query,
                "answer": "An internal error occurred.",
                "source_documents": []
            }


# Initialize pipeline instance
pipeline = RetrievalPipeline()
