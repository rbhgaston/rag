# RAG Retrieval Augmented Generation
## let ai search your files and get you answers for your questions

1. Clone the repo
2. Create your virtual env
3. install depenacies from requiremenets.txt

Then you have two options:
### Terminal interface
First open create_db.py file and change the DATA_PATH and CHROMA_PATH.
You can change the embeddings too.
Pay attention to seq length when changing the embeddings model.

Run create_db.py file to create the database. Rerun the file only to update the database.
It accepts only pdf files.


Run main.py after specifying the query to the llm.

### Streamlit interface
run streamlit run app.py



### FEATURES
you can have many chunk sizes

### COMMENTS
db.similarity_search_with_relevance_scores gives better results that other similiraties functions.

Importance of prompting also: giving clear and concise instructions yields the best results.

rag works best when giving response to a specific question but not a good at summary like: give the factors of an increased inflation?

instead of storing diffrent embedded chunk size in the same db, we can create a db for each one 
because when querying for respones, the model tends to use only the small sized chunks.

mxbai-embed-large: seq len: 512
nomic: seq len: 8196 

GOT THE BEST RES WITH MXBAI AND CHUNK_SIZE = 256 512 1024 2048 8192

