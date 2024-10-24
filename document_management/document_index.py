from elasticsearch_dsl import Document, Text, connections, Search

# Establish a connection to Elasticsearch
connections.create_connection(hosts=['http://localhost:9211'])

# Define a Document schema for Elasticsearch
class DocumentIndex(Document):
    filename = Text()
    content = Text()

    class Index:
        name = 'documents'  # Name of the index in Elasticsearch

# Initialize the index (create it in Elasticsearch)
DocumentIndex.init()

# Function to save a document to Elasticsearch
def save_document_to_elasticsearch(filename, content):
    doc = DocumentIndex(filename=filename, content=content)
    doc.save()

# Function to search for documents by keyword
def search_documents(keyword):
    s = Search(index="documents").query("match", content=keyword)
    response = s.execute()
    return response
