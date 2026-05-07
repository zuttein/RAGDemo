import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq
from dotenv import load_dotenv
import os

# Dummy data 
texts = [
    "Semester ansöks via HR-systemet.",
    "Arbetstid är 40 timmar per vecka.",
    "Lön betalas ut i efterskott.",
    "Sjukfrånvaro rapporteras i HR-systemet.",
    """
    Multifaktorautentisering, även kallat MFA, ska användas för system som innehåller känslig
    information eller ger åtkomst till interna resurser. MFA minskar risken för obehörig åtkomst
    även om ett lösenord skulle bli känt.
    """,
    """
    Om en medarbetare misstänker att ett lösenord har läckt ska lösenordet bytas omedelbart
    och incidenten rapporteras till IT-supporten. Vid misstänkt intrång kan kontot tillfälligt
    spärras medan IT utreder händelsen.
    """,
    """
    Timanställda rapporterar arbetade timmar i tidrapporteringssystemet. Timmarna måste vara
    attesterade av ansvarig chef innan de kan ligga till grund för löneutbetalning.
    """,
    """
    Lön betalas ut månadsvis i efterskott, normalt den 25:e varje månad. Om den 25:e infaller
    på en helgdag sker utbetalningen vanligtvis närmast föregående bankdag.
    """,

    """
Informationssäkerhet handlar om att skydda information så att den bara är tillgänglig
för rätt personer, vid rätt tillfälle och på rätt sätt. Inom företag används ofta tre
grundläggande principer: konfidentialitet, integritet och tillgänglighet. Konfidentialitet
betyder att obehöriga personer inte ska kunna läsa informationen. Integritet betyder att
informationen ska vara korrekt och inte ändras av misstag eller av någon obehörig.
Tillgänglighet betyder att information och system ska vara åtkomliga när verksamheten
behöver dem.

Ett vanligt hot mot informationssäkerhet är phishing. Phishing innebär att en angripare
försöker lura en användare att klicka på en falsk länk, öppna en skadlig bilaga eller
skriva in sina inloggningsuppgifter på en falsk webbplats. Sådana attacker kan se mycket
trovärdiga ut och kan till exempel låtsas komma från en chef, en bank, en myndighet eller
en intern IT-avdelning. Därför är det viktigt att användare kontrollerar avsändare,
länkar och språk innan de agerar på ett oväntat meddelande.

Lösenord är fortfarande en central del av många säkerhetssystem, men lösenord ensamma
räcker sällan för att ge ett starkt skydd. Ett bra lösenord bör vara långt, unikt och
svårt att gissa. Det bör inte återanvändas mellan privata och arbetsrelaterade konton.
Om samma lösenord används på flera platser kan ett läckt lösenord från en tjänst användas
för att angripa andra tjänster. Därför rekommenderas ofta lösenordshanterare som kan
skapa och lagra starka lösenord åt användaren.

Multifaktorautentisering, ofta förkortat MFA, innebär att användaren måste bevisa sin
identitet med mer än en faktor. Det kan till exempel vara något användaren vet, som ett
lösenord, kombinerat med något användaren har, som en mobiltelefon eller säkerhetsnyckel.
MFA minskar risken för obehörig åtkomst eftersom ett stulet lösenord inte räcker för att
logga in. För system som hanterar känslig information eller interna resurser bör MFA vara
aktiverat som standard.

Behörighetsstyrning är också viktigt. Alla användare bör bara ha den åtkomst de faktiskt
behöver för sitt arbete. Detta kallas principen om minsta privilegium. Om en användare
har för bred behörighet kan skadan bli större om kontot kapas eller används felaktigt.
När en medarbetare byter roll eller slutar på företaget bör behörigheter därför granskas
och uppdateras. Gamla konton som inte längre används bör stängas så snabbt som möjligt.

Säkerhetsincidenter ska rapporteras snabbt. En incident kan vara ett misstänkt intrång,
förlorad dator, skickad information till fel mottagare, skadlig kod, ovanlig aktivitet på
ett konto eller ett klick på en misstänkt länk. Syftet med snabb rapportering är inte att
skuldbelägga användaren, utan att begränsa skadan och ge IT eller säkerhetsansvariga
möjlighet att agera i tid. Ju tidigare en incident rapporteras, desto större chans finns
det att förhindra större konsekvenser.

Säkerhetskopiering är en annan viktig del av informationssäkerhet. Om data försvinner på
grund av tekniska fel, mänskliga misstag eller ransomware behöver organisationen kunna
återställa informationen. Backuper bör tas regelbundet, lagras säkert och testas med jämna
mellanrum. En backup som aldrig testats kan ge en falsk trygghet, eftersom organisationen
inte vet om den faktiskt går att använda vid en kris.

Informationssäkerhet är därför inte bara en teknisk fråga. Det handlar också om rutiner,
utbildning, ansvar och kultur. Teknik som brandväggar, antivirus, kryptering och MFA är
viktig, men användarnas beteende spelar också stor roll. En organisation med god
säkerhetskultur gör det lätt för medarbetare att göra rätt, rapportera misstag och följa
tydliga processer utan onödig friktion.
"""

]

@st.cache_resource # Spara db i cache
def chroma_setup():
    # Skapar en text-splitter
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    
    #Omvandlar text till dokument object (Varje text blir ett dokument)
    docs = splitter.create_documents(texts)
    #st.write("Antal dokument:", len(docs))
    
    # Skapar en embedding-modell gör om mina dokument till vektorer
    embeddings = HuggingFaceEmbeddings()

    return Chroma.from_documents(docs, embeddings)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Alla "dokument" blir till embeddings och sparas
db = chroma_setup()

# UI
st.title("Retrieval Demo")

# UI
query = st.text_input("Ställ en fråga")

# Retrieval - R 
#När användaren skriver något körs retrieval
if query:
    #Frågan blir embedding och jämförs med alla "dokument", räknar likheter och returnerar den bästa matchen
    results = db.similarity_search(query, k=3)

    #  Context - A 

    # Sätt ihop alla dokument till en text som ska till AI
    context = "\n\n".join([doc.page_content for doc in results ])

    #  LLM - G 

    prompt = f"""
        Context: {context}

        Fråga: {query}

        Ge ditt svar:
    """

    response = client.chat.completions.create( # Skapa en klient från Groq. Specifiera modell och instruktioner till AI samt prompt
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """
                            Du är en RAG-assistent.
                            Regler:
                            1. Svara enbart baserat på informationen i context.
                            2. Om svaret inte finns i context svara då: "Efterfrågad information finns inte i dokumenten"
                            3. Hitta inte på information.
                            4. Svara tydligt och kort.
                            """
            },
            { 
                "role": "user",
                "content": prompt
            }
            ],
            temperature=0.3
    )

    answer = response.choices[0].message.content
    st.write(answer)




