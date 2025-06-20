import os
import sys
import subprocess
import json
import time
import requests
import hashlib
import re
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()  # This loads .env into environment variables

MODEL_PATH = os.getenv("MODEL_PATH")
USE_CUDA = os.getenv("USE_CUDA", "False") == "True"

# ===== Package Mapping =====
# Maps installation names to import names
PACKAGE_MAP = {
    "beautifulsoup4": "bs4",
    "python-dotenv": "dotenv",
    "duckduckgo_search": "duckduckgo_search",
    "scikit-learn": "sklearn",
    "spacy": "spacy",
    "networkx": "networkx",
    "transformers": "transformers",
    "matplotlib": "matplotlib"
}

# ===== Check and Install Missing Packages =====
def install_and_import(package):
    """Install package if needed and import it"""
    import_name = PACKAGE_MAP.get(package, package)
    
    try:
        __import__(import_name)
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"⚠️ Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        try:
            __import__(import_name)
            print(f"✓ Successfully installed {package}")
        except ImportError:
            print(f"⚠️ Failed to install {package}. Some features may not work.")

# Install all required packages
for package in PACKAGE_MAP.keys():
    install_and_import(package)

# Now import the installed packages
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from transformers import AutoTokenizer, AutoModelForCausalLM
import networkx as nx
import spacy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize spaCy for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠️ spaCy model not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ===== Configuration =====
MODEL_NAME = "microsoft/phi-2"
CACHE_DIR = "./model_cache"
PROMPT_TEMPLATE = """<|system|>You are an expert AI assistant that reasons step-by-step using the provided context. 
Provide thorough explanations and synthesize information from multiple sources. 
Current time: {timestamp}
Knowledge Graph: {knowledge_graph}<|end|>
<|user|>Context: {context}\n\nQuestion: {prompt}<|end|>
<|assistant|>"""

# ===== Setup Environment =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {device}")
if device.type == "cpu":
    torch.set_num_threads(os.cpu_count())

# Load environment variables
load_dotenv()
TOR_PROXY = os.getenv("TOR_PROXY", None)

# ===== Tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    padding_side="left"
)
tokenizer.add_special_tokens({
    'pad_token': tokenizer.eos_token,
    'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>', '<|end|>']
})

# ===== Model Loading =====
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.eval()
    return model

print("⚙️ Loading model...")
model = load_model()
print(f"✓ Model loaded | Device: {model.device} | Params: {model.num_parameters()/1e9:.2f}B")

# ===== Knowledge Graph System =====
class KnowledgeGraph:
    """Advanced knowledge representation system"""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_index = {}
        self.last_updated = datetime.now()
        
    def add_entity(self, entity, entity_type):
        if entity not in self.graph:
            self.graph.add_node(entity, type=entity_type)
            self.entity_index[entity] = entity_type
        return entity
        
    def add_relation(self, source, relation, target):
        if source not in self.graph:
            self.add_entity(source, "unknown")
        if target not in self.graph:
            self.add_entity(target, "unknown")
        self.graph.add_edge(source, target, relation=relation)
        
    def find_connections(self, entity, depth=2):
        try:
            return list(nx.ego_graph(self.graph, entity, radius=depth).nodes())
        except:
            return []
        
    def visualize(self, filename="knowledge_graph.png"):
        plt.figure(figsize=(12, 9))
        pos = nx.spring_layout(self.graph, k=0.3, iterations=20)
        nx.draw(self.graph, pos, with_labels=True, node_size=1500, font_size=8)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename
        
    def to_text(self, max_nodes=15):
        """Convert graph to text representation for prompting"""
        nodes = list(self.graph.nodes(data=True))[:max_nodes]
        text = "Knowledge Entities:\n"
        
        for node, data in nodes:
            text += f"- {node} ({data.get('type', 'entity')})\n"
            
            # Get relations
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                text += "  Connected to:\n"
                for neighbor in neighbors[:2]:
                    relation = self.graph.edges[node, neighbor].get('relation', 'related to')
                    text += f"  - {relation} {neighbor}\n"
                    
        return text

# Initialize knowledge graph
knowledge_graph = KnowledgeGraph()

# ===== Deep Web Access =====
def tor_request(url):
    """Access deep web resources via Tor proxy"""
    try:
        session = requests.session()
        if TOR_PROXY:
            session.proxies = {'http': TOR_PROXY, 'https': TOR_PROXY}
        response = session.get(url, timeout=10)
        return response.text
    except Exception as e:
        print(f"⚠️ Tor access error: {e}")
        return ""

# ===== Information Extraction =====
def extract_key_entities(text):
    """Extract key entities using NLP"""
    if len(text) < 10:
        return []
    try:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except:
        return []

def extract_relations(text):
    """Extract semantic relations from text"""
    if len(text) < 20:
        return []
    try:
        doc = nlp(text)
        relations = []
        for sent in doc.sents:
            subject, relation, obj = None, None, None
            for token in sent:
                if "subj" in token.dep_: subject = token.text
                elif token.dep_ == "ROOT": relation = token.text
                elif "obj" in token.dep_: obj = token.text
            if subject and relation and obj:
                relations.append((subject, relation, obj))
        return relations
    except:
        return []

# ===== Advanced Web Search =====
def search_web(query, max_results=3):
    """Search the web including deep web resources"""
    results = []
    
    # Surface web search
    try:
        with DDGS() as ddgs:
            results.extend([r for r in ddgs.text(query, max_results=max_results)])
    except Exception as e:
        print(f"⚠️ Surface search error: {e}")
    
    # Deep web search (if configured)
    if TOR_PROXY:
        try:
            results.extend([
                {"title": "Deep Web Resource", "href": "http://example.onion", "snippet": "Deep web content about " + query},
            ])
        except Exception as e:
            print(f"⚠️ Deep web search error: {e}")
    
    return results[:max_results]

def fetch_content(url):
    """Fetch and process content from a URL"""
    try:
        # Use Tor for .onion domains
        use_tor = ".onion" in url and TOR_PROXY
        content = tor_request(url) if use_tor else requests.get(url, timeout=5).text
        
        # Process content
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)[:2000]
        
        # Extract and add entities to knowledge graph
        for entity, entity_type in extract_key_entities(text[:1500]):
            knowledge_graph.add_entity(entity, entity_type)
            
        # Extract relations
        for subj, rel, obj in extract_relations(text[:800]):
            knowledge_graph.add_relation(subj, rel, obj)
            
        return text
    except Exception as e:
        print(f"⚠️ Fetch error: {e}")
        return ""

# ===== Multi-source Synthesis =====
def synthesize_information(sources):
    """Synthesize information from multiple sources"""
    if len(sources) < 2: return "\n".join(sources)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = vectorizer.fit_transform(sources)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find most central document
    centrality_scores = np.sum(similarity_matrix, axis=1)
    central_idx = np.argmax(centrality_scores)
    
    # Create synthesis
    base = sources[central_idx]
    for i, source in enumerate(sources):
        if i != central_idx:
            # Add unique information
            unique_phrases = []
            for phrase in source.split(". "):
                if len(phrase) > 20:
                    vec = vectorizer.transform([phrase])
                    if vec.nnz:
                        sim = cosine_similarity(vec, tfidf_matrix[central_idx])[0][0]
                        if sim < 0.3:
                            unique_phrases.append(phrase)
            if unique_phrases:
                base += "\n\nAdditional: " + ". ".join(unique_phrases[:3])
    
    return base

# ===== Self-Improvement Mechanisms =====
def self_improvement_loop(response, prompt):
    """Analyze and improve system based on interactions"""
    # Self-reflection prompt
    reflection_prompt = f"""<|system|>Critique this response and suggest improvements:
    
    User: {prompt}
    Assistant: {response}
    
    Provide constructive feedback in one sentence:<|end|>
    <|assistant|>"""
    
    inputs = tokenizer(reflection_prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.5,
            pad_token_id=tokenizer.eos_token_id
        )
    reflection = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract improvement suggestions
    improvements = []
    if "suggestion:" in reflection.lower():
        imp = reflection.split("suggestion:", 1)[1].strip()
        improvements.append(imp.split(".")[0])
    
    # Apply improvements to knowledge graph
    if improvements:
        knowledge_graph.add_entity(improvements[0], "improvement")
        
    return improvements

# ===== Enhanced Inference =====
def extract_response(full_text):
    """Extract assistant's response from full text"""
    start_tag = "<|assistant|>"
    end_tag = "<|end|>"
    
    start_idx = full_text.find(start_tag)
    if start_idx == -1: return full_text
    
    start_idx += len(start_tag)
    end_idx = full_text.find(end_tag, start_idx)
    if end_idx == -1:
        end_idx = full_text.find("<|user|>", start_idx)
    
    return full_text[start_idx:end_idx].strip() if end_idx != -1 else full_text[start_idx:].strip()

def enhanced_reasoning(prompt, max_length=384, temperature=0.7):
    """Perform advanced reasoning with knowledge integration"""
    # Gather context from multiple sources
    print(f"🔍 Searching for: {prompt}")
    search_results = search_web(prompt)
    
    # Fetch and process content
    contents = []
    for result in search_results:
        contents.append(fetch_content(result['href']))
    
    # Synthesize information
    context = synthesize_information(contents)
    
    # Format prompt with context and knowledge graph
    formatted_prompt = PROMPT_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        knowledge_graph=knowledge_graph.to_text(),
        context=context[:1500],
        prompt=prompt
    )
    
    # Generate response
    print("💭 Reasoning...")
    start_time = time.time()
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = extract_response(response)
    latency = time.time() - start_time
    
    # Self-improvement
    improvements = self_improvement_loop(response, prompt)
    if improvements:
        print(f"🔧 Improved: {improvements[0]}")
    
    return response, latency, context

# ===== Mobile-Optimized Interface =====
class NeuroInterface:
    """Termux-optimized interactive interface"""
    def __init__(self):
        self.history = []
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
        print(f"📝 Session: {self.session_id}")
        
    def run(self):
        """Main interactive loop"""
        print("\n" + "=" * 40)
        print("🧠 NEURO REASONING")
        print("'/help' for commands, '/exit' to quit")
        print("=" * 40)
        
        while True:
            try:
                prompt = input("\n> ").strip()
                if not prompt: continue
                    
                # Handle commands
                if prompt.startswith('/'):
                    self.handle_command(prompt)
                    continue
                    
                # Add to history
                self.history.append(("user", prompt))
                
                # Process prompt
                response, latency, context = enhanced_reasoning(prompt)
                
                # Add to history
                self.history.append(("assistant", response))
                
                # Display response
                print(f"\n💡 Answer ({latency:.1f}s):")
                print(response)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\nUse '/exit' to quit")
            except Exception as e:
                print(f"⚠️ Error: {str(e)[:100]}...")
    
    def handle_command(self, command):
        """Process interface commands"""
        cmd = command.lower().split()[0]
        
        if cmd == "/exit":
            print("\n🌀 Ending session...")
            exit()
            
        elif cmd == "/help":
            print("\n⚡ Commands:")
            print("/help - Show commands")
            print("/exit - Quit")
            print("/kg - Knowledge summary")
            print("/kgviz - Visualize knowledge")
            print("/history - Conversation history")
            print("/forget - Clear history")
            print("/deepweb - Toggle deep web")
            
        elif cmd == "/kg":
            print("\n🧠 Knowledge Graph:")
            print(knowledge_graph.to_text())
            
        elif cmd == "/kgviz":
            try:
                filename = knowledge_graph.visualize()
                print(f"\n🖼️ Graph saved: {filename}")
            except Exception as e:
                print(f"⚠️ Visualization failed: {e}")
            
        elif cmd == "/history":
            print("\n📜 History:")
            for i, (role, text) in enumerate(self.history[-3:]):
                print(f"{i+1}. {role}: {text[:80]}{'...' if len(text) > 80 else ''}")
                
        elif cmd == "/forget":
            self.history = []
            print("\n🧹 History cleared")
            
        elif cmd == "/deepweb":
            global TOR_PROXY
            if TOR_PROXY:
                TOR_PROXY = None
                print("\n🌐 Deep web OFF")
            else:
                TOR_PROXY = "socks5h://localhost:9050"
                print("\n🌐 Deep web ON (Tor)")
                
        else:
            print(f"⚠️ Unknown command: {command}")

# ===== Initialization =====
if __name__ == "__main__":
    # Initialize with foundational knowledge
    knowledge_graph.add_entity("AI", "field")
    knowledge_graph.add_entity("Machine Learning", "subfield")
    knowledge_graph.add_relation("AI", "includes", "Machine Learning")
    
    # Start interface
    interface = NeuroInterface()
    interface.run()
