Set Up Core Azure Resources 🏗️
Even with a code-centric LangChain approach, you'll need these foundational Azure services. You can create them via the Azure portal:
Azure OpenAI Service:
Deployments: Once created, go to Azure OpenAI Studio. You'll need to deploy:
A Chat Model: e.g., gpt-35-turbo-16k or gpt-4. Note your deployment name.
An Embedding Model: e.g., text-embedding-ada-002. Note your deployment name.
Credentials: Note down your Azure OpenAI endpoint, API key, and the deployment names.
Azure AI Search Service:


Azure Blob Storage Account (Optional but Recommended):
Creation: In the Azure portal, search for "Storage accounts" and create one. Create containers within it as needed (e.g., raw-documents, source-code, logs).
Credentials: Note your storage account connection string if you plan to use it with LangChain's Azure Blob Storage loaders.

