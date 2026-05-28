FROM python:3.12

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y curl

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -

RUN apt-get install -y nodejs

RUN npx playwright install chrome

RUN pip install --default-timeout=1000 -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]