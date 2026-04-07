"""
vedology_engine.py — Clean Industry-Level Vedology Engine
"""

from retriever import Retriever
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:1.7b"


class VedologyEngine:

    def __init__(self):
        print("🔧 Initializing Vedology Engine...")
        self.retriever = Retriever()
        print("✅ Engine Ready\n")

    # 🔥 STEP 1: Check if query is relevant
    def is_valid_context(self, context: str) -> bool:
        if "No relevant passages found" in context:
            return False
        if len(context.strip()) < 80:
            return False
        return True

    # 🔥 STEP 2: Build strong prompt
    def build_prompt(self, query: str, context: str, lang: str) -> str:

        if lang == "hindi":
            return f"""
आप Vedology हैं — एक सटीक और भरोसेमंद AI।

नियम:
- केवल दिए गए context का उपयोग करें
- अगर उत्तर नहीं है → साफ लिखें: "उत्तर उपलब्ध नहीं है"
- कोई अनुमान नहीं

FORMAT:

1. छोटा उत्तर (1-2 लाइन)
2. सरल व्याख्या
3. वास्तविक भारतीय उदाहरण (strong, relatable)
4. स्रोत

User Question:
{query}

Context:
{context}

Answer:
"""
        else:
            return f"""
You are Vedology — a precise, wise, and deeply insightful AI.

You explain philosophy in a structured, practical, and very clear way.

STRICT RULES:
- Use ONLY the given context
- If answer is not present → say: "No relevant answer found"
- Do NOT guess
- Be clear, structured, and impactful

TONE:
- You are a calm, intelligent, experienced Indian guide
- Speak with clarity and purpose
- No unnecessary words

FORMAT (STRICT — follow exactly):

1. Detailed Answer:
(A complete and well-formed answer covering the core idea clearly)

2. Short Answer:
(1-2 lines maximum, very direct)

3. Detailed Explanation:
(Break down the idea in a simple but deep way)

4. Real-Life Indian Analogy:
(Use strong, relatable Indian example — family, job, studies, society)

5. Source:
(Mention book name or text clearly)

6. Steps to Implement in Real Life:
(3-5 clear, actionable steps)

User Question:
{query}

Context:
{context}

Answer:
"""

    # 🔥 STEP 3: Call local LLM
    def call_llm(self, prompt: str) -> str:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
        )

        if response.status_code != 200:
            return "LLM Error"

        return response.json().get("response", "")

    # 🔥 STEP 4: Main answer pipeline
    def answer(self, query: str, lang: str) -> str:

        context = self.retriever.retrieve_for_llm(query)

        # 🚫 Reject bad queries
        if not self.is_valid_context(context):
            if lang == "hindi":
                return "🧠 Vedology:\n\nइस प्रश्न का उत्तर उपलब्ध ज्ञान में नहीं है।"
            else:
                return "🧠 Vedology:\n\nNo relevant answer found in the knowledge base."

        prompt = self.build_prompt(query, context, lang)

        response = self.call_llm(prompt)

        return "\n🧠 Vedology:\n\n" + response.strip()


# 🔥 CLI Interface
def main():

    engine = VedologyEngine()

    print("=== Vedology (Industry Engine) ===")
    print("Type 'exit' to quit\n")

    while True:

        lang = input("🌐 Choose language (english/hindi): ").strip().lower()

        if lang not in ["english", "hindi"]:
            print("❌ Invalid language\n")
            continue

        query = input("You: ")

        if query.lower() == "exit":
            break

        answer = engine.answer(query, lang)

        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()