# Use an official Python image as the base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app


#Set environment variables
ENV TAVILY_API_KEY=
ENV LANGCHAIN_TRACING_V2=
ENV LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
ENV LANGCHAIN_API_KEY=



# Copy your Python scripts into the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Create the folder inside the container
RUN mkdir -p /app/pdfs

COPY /pdfs/*.pdf /app/pdfs


# Expose the port your FastAPI app will run on
EXPOSE 8005
#EXPOSE 11434

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
