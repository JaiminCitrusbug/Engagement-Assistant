import os
from openai import OpenAI
from retriever import retrieve_similar_chunks

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def determine_retrieval_query(user_query, history):
    """
    Use LLM to dynamically determine the best query for retrieval based on conversation context.
    The LLM analyzes the conversation and determines what should be searched in the vector database.
    """
    # Convert history to chat format (skip system messages)
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history if m.get("role") != "system"]
    
    # Create a prompt for the LLM to determine the retrieval query
    query_determination_prompt = {
        "role": "system",
        "content": """You are a query analyzer for an engagement and campaign management information retrieval system. 
Your task is to analyze the conversation and determine the BEST search query to use for retrieving relevant information from the knowledge base.

CRITICAL RULES:
1. If the user is asking for MORE INFORMATION about something previously discussed (e.g., "more info", "tell me more", "more details", "what else"):
   - Look at the conversation history to identify what was discussed in previous messages
   - Extract the topic (campaign type, metric, strategy, etc.) that was mentioned earlier
   - Use that extracted term as the search query
   - Example: If user previously asked about "email campaigns" and now says "more info", return "email campaigns"
   - Example: If user previously asked about "segmentation" and now says "tell me more", return "segmentation targeting"

2. If the user is asking about campaigns or channels:
   - Extract campaign types (email, SMS, social media, webinar, push notifications)
   - Use terms like "email campaigns", "SMS campaigns", "social media campaigns", "webinar campaigns"
   - Example: "how to run email campaigns" → Return: "email campaigns"
   - Example: "best practices for SMS" → Return: "SMS campaigns"

3. If the user is asking about segmentation or targeting:
   - Use terms like "segmentation", "targeting", "audience segmentation", "HCP segmentation", "patient segmentation"
   - Example: "how to segment HCPs" → Return: "HCP segmentation targeting"
   - Example: "targeting strategies" → Return: "targeting strategies segmentation"

4. If the user is asking about content automation:
   - Use terms like "content automation", "automated content", "content workflows", "automation"
   - Example: "how to automate content delivery" → Return: "content automation"
   - Example: "welcome series workflow" → Return: "welcome series automation workflow"

5. If the user is asking about tracking or metrics:
   - Use terms like "campaign tracking", "metrics", "effectiveness tracking", "ROI", "engagement metrics"
   - Example: "how to track campaign performance" → Return: "campaign effectiveness tracking metrics"
   - Example: "what metrics should I track" → Return: "campaign metrics tracking"

6. If the user is asking about doctor journey or HCP journey:
   - Use terms like "doctor journey", "HCP journey", "journey mapping", "journey stages"
   - Example: "stages of doctor journey" → Return: "doctor journey mapping stages"
   - Example: "how to engage HCPs at different stages" → Return: "doctor journey engagement"

7. If the user is referring to something mentioned earlier (using pronouns like "it", "that", "this"), extract the actual topic from the conversation history.

8. Always return ONLY the search query term(s) - no explanations, no questions, just the query string.

IMPORTANT: For follow-up queries like "more info", ALWAYS extract the topic from conversation history - never return the follow-up phrase itself.
IMPORTANT: Be specific - include campaign types, metric names, or strategy keywords.

Return ONLY the search query, nothing else."""
    }
    
    # Build messages for query determination
    messages = [query_determination_prompt] + chat_history + [
        {"role": "user", "content": f"Given the conversation above, what should be the search query for the current user message: '{user_query}'?\n\nReturn ONLY the search query:"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=50,
        )
        
        retrieval_query = response.choices[0].message.content.strip()
        
        # Clean up the response
        retrieval_query = retrieval_query.strip('"').strip("'").strip()
        
        if ":" in retrieval_query and len(retrieval_query.split(":")) > 1:
            retrieval_query = retrieval_query.split(":")[-1].strip()
        
        if not retrieval_query or len(retrieval_query) < 2:
            retrieval_query = user_query
        
        print(f"🤖 LLM determined retrieval query: '{retrieval_query}' (from user: '{user_query}')")
        return retrieval_query
        
    except Exception as e:
        print(f"⚠️  Error in LLM query determination: {e}. Using original query.")
        return user_query

def create_system_prompt(context_text):
    """Create the system prompt for the engagement assistant."""
    return {
        "role": "system",
        "content": (
            "You are an expert Engagement and Campaign Management Assistant. "
            "Your role is to help with multi-channel campaigns, segmentation & targeting, content automation, "
            "campaign effectiveness tracking, and doctor journey mapping. "
            "Use the entire conversation history and the provided context to give accurate, actionable, and strategic answers.\n\n"

            "CRITICAL: Response Style - Be CONCISE, CONFIDENT, and DIRECT:\n"
            "- Keep responses SHORT (2-4 sentences maximum, unless user explicitly asks for more)\n"
            "- Be confident and direct - provide the essential information needed to answer the question\n"
            "- Provide clear, practical guidance based on best practices, but keep it brief\n"
            "- Do NOT add unnecessary explanations or verbose elaborations\n"
            "- Do NOT list multiple strategies unless the question asks for them\n"
            "- Do NOT dump raw data - provide concise, well-structured summaries\n"
            "- Be strategic and actionable, but keep it short\n\n"

            "Response Guidelines (CONCISE):\n"
            "- For campaign questions: Give key best practice or template in 2-3 sentences. Details only if asked.\n"
            "- For segmentation questions: State the segmentation approach in 2-3 sentences. Full criteria list only if asked.\n"
            "- For content automation: Explain the workflow or trigger in 2-3 sentences. Full setup only if asked.\n"
            "- For tracking questions: State the metric and benchmark in 2-3 sentences. Full calculation details only if asked.\n"
            "- For journey mapping: Briefly explain the stage or tactic in 2-3 sentences. Full details only if asked.\n\n"

            "ELABORATION RULES:\n"
            "- ONLY elaborate when user explicitly asks: 'more info', 'more details', 'tell me more', 'elaborate', 'explain more', 'give me more information', 'expand', 'detailed', 'full details', 'complete information'\n"
            "- When user asks to elaborate, THEN provide additional relevant information, examples, templates, or detailed strategies\n"
            "- Default response should be SHORT and DIRECT - save detailed explanations for when explicitly requested\n\n"

            "Data and Metrics (CONCISE):\n"
            "- When discussing metrics, provide the key metric name and benchmark in 1-2 sentences\n"
            "- Only include calculations, improvement tips, and detailed strategies when user asks to elaborate\n"
            "- Reference best practices briefly - full details only if requested\n\n"

            f"Context from knowledge base:\n{context_text}\n\n"
            "Remember: Be CONCISE and DIRECT. Provide short, confident answers (2-4 sentences). "
            "Only elaborate when the user explicitly asks for more information. "
            "Be specific and practical, but keep it brief. If information is not available in the context, say so briefly."
        ),
    }

def generate_answer(user_query, history):
    """Generate context-aware answer using chat history and RAG."""
    # Check if user is asking for more information (increase top_k for more comprehensive retrieval)
    user_query_lower = user_query.lower().strip()
    is_more_data_request = any(phrase in user_query_lower for phrase in [
        "more info", "more information", "more details", "tell me more", 
        "what else", "anything else", "additional", "more about",
        "elaborate", "explain more", "give me more", "expand",
        "detailed", "full details", "complete information"
    ])
    
    # Use LLM to dynamically determine the best query for retrieval
    retrieval_query = determine_retrieval_query(user_query, history)
    
    # Retrieve chunks - use more chunks if user asks for "more info"
    top_k = 10 if is_more_data_request else 5
    retrieved_chunks = retrieve_similar_chunks(retrieval_query, top_k=top_k)
    context_text = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    
    # If no context retrieved, still proceed but with empty context
    if not context_text.strip():
        context_text = "No relevant information found in the knowledge base for this query."
    
    # Convert Streamlit history to OpenAI chat format (exclude system messages from history)
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history if m.get("role") != "system"]
    system_prompt = create_system_prompt(context_text)
    
    messages = [system_prompt] + chat_history
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"⚠️  Error generating answer: {e}")
        return "Sorry, I'm having trouble responding right now. Please try again."

