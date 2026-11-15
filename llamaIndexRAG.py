
from llama_index.core import SimpleDirectoryReader

# We use simple directory reader to load docs of different types
our_reader = SimpleDirectoryReader(input_dir="/Users/matthew/Documents/elevenlabs")
documents = our_reader.load_data()
print(f"Loaded {len(documents)} documents from the directory.")