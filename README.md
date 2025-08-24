# RAG-AzureAI-Module

This repository contains a comprehensive production-level multimodal RAG service using Azure AI.

## File Structure

```
RAG-AzureAI-Module/
│
├── src/
│   ├── __init__.py
│   ├── main.py                # Entry point for the application
│   ├── config.py             # Configuration settings
│   ├── azure_services/
│   │   ├── __init__.py
│   │   ├── openai_service.py  # Interactions with Azure OpenAI
│   │   ├── search_service.py   # Interactions with Azure Cognitive Search
│   │   ├── document_intelligence.py  # Document processing with Azure Document Intelligence
│   │   ├── computer_vision.py  # Image processing with Azure Computer Vision
│   │
│   ├── rAG_service.py         # Main RAG service logic
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Load data for processing
│   │   ├── preprocess.py        # Preprocessing functions for text/images
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          # Logging setup
│       └── helpers.py         # Helper functions
│
├── tests/
│   ├── __init__.py
│   ├── test_rag_service.py    # Unit tests for the RAG service
│   ├── test_azure_services.py  # Unit tests for Azure service interactions
│   └── test_utils.py          # Unit tests for utility functions
│
├── docker/
│   ├── Dockerfile              # Dockerfile for application
│   └── docker-compose.yml      # Docker Compose configuration
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file
```
