from prompts import PLANNER_PROMPT

# Funktion som använder en LLM som planner
# Tar emo:
# Client = Groq-klienten för att prata med modellen
# Query = användarens fråga
def planner_llm(client, query):
    
    # Stoppar in användarens fråga i planner-prompten
    #{query} i prompten ersätts med själva frågan
    planner_prompt = PLANNER_PROMPT.format(
        query=query)
    
    #Skickar prompten till LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        
        #Konversationen till modellen
        messages=[
            {
                "role": "system",
                #Specifierar att modellen är en strikt planner
                "content": (
                    "Du är en strikt planner"
                    "för en RAG-applikation."
                
                )
            }, 
            {
                "role": "user",
                "content": planner_prompt
            }
        ],
        #Gör modellen stabil, plannern ska inte vara kreativ
        temperature=0.0
        
        )
    #Hämtar svaret från modellen, alltså rag eller irrelevant
    route = (
        response
    .choices[0]
    .message.content
    .strip()
    .lower()
    )
    
    #Debug
    print(f"""
          Planner query: {query}
          planner response: {route}
          """)
    
    
    return route