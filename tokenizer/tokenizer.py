import json
import os
import re
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
from pathlib import Path
import unicodedata
import hashlib
import random
import logging
from typing import List, Dict, Tuple, Union, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='⚛️ [%(levelname)s] %(message)s')
logger = logging.getLogger("NeuroQuantumTokenizer")

# Constants
VOCAB_FILE = "vocab.json"
CACHE_DIR = "search_cache"
NEURAL_MEMORY_FILE = "neural_memory.json"
QUANTUM_SEQUENCES_FILE = "quantum_sequences.json"
MAX_VOCAB_SIZE = 65536  # 2^16 for better performance
MAX_CACHE_AGE = 86400  # 1 day in seconds

class NeuroQuantumTokenizer:
    def __init__(self, mode: str = 'hybrid', vocab_file: str = VOCAB_FILE):
        """
        Ultimate Neuro-Quantum Tokenizer with quantum-enhanced capabilities
        Features:
        - Hybrid tokenization (char + subword + neural patterns)
        - Quantum-inspired encoding sequences
        - Neural memory for pattern recognition
        - Web lookup with contextual understanding
        - Dynamic vocabulary expansion
        - Multi-engine search with caching
        - State serialization/deserialization
        """
        self.mode = mode
        self.vocab_file = vocab_file
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize all components
        self.special_tokens = self._init_special_tokens()
        self.unk_token_id = self.special_tokens.get("<|unk|>", 1)
        self.pad_token_id = self.special_tokens.get("<|pad|>", 0)
        self.bos_token_id = self.special_tokens.get("<|bos|>", 15)
        self.eos_token_id = self.special_tokens.get("<|eos|>", 16)
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.neural_memory: Dict[str, List[str]] = {}
        self.quantum_sequences: Dict[str, str] = {}
        self.search_engines = self._init_search_engines()
        self.patterns = self._init_patterns()
        
        # Load data
        self.load_vocab()
        self.load_neural_memory()
        self.load_quantum_sequences()
        
        # Initialize with basic characters if needed
        if len(self.stoi) == len(self.special_tokens):
            self._init_basic_vocab()

    def _init_special_tokens(self) -> Dict[str, int]:
        """Special tokens for neural and quantum operations"""
        return {
            "<|pad|>": 0,
            "<|unk|>": 1,
            "<|start|>": 2,
            "<|end|>": 3,
            "<|neuro|>": 4,
            "<|quantum|>": 5,
            "<|memory|>": 6,
            "<|reason|>": 7,
            "<|url|>": 8,
            "<|num|>": 9,
            "<|email|>": 10,
            "<|phone|>": 11,
            "<|entity|>": 12,
            "<|sep|>": 13,
            "<|mask|>": 14,
            "<|bos|>": 15,
            "<|eos|>": 16
        }

    def _init_patterns(self) -> Dict[str, re.Pattern]:
        """Regex patterns for neural processing"""
        return {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*\??[^ \n]*'),
            'number': re.compile(r'\b\d+\.?\d*\b'),
            'phone': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            'special_seq': re.compile(r'<\|[a-z]+\|>'),
            'word': re.compile(r'\w+|[^\w\s]'),
            'html_tag': re.compile(r'<[^>]+>'),
            'hashtag': re.compile(r'#\w+'),
            'mention': re.compile(r'@\w+')
        }

    def _init_search_engines(self) -> Dict[str, Callable]:
        return {
            "google": self._google_search,
            "bing": self._bing_search,
            "ddg": self._duckduckgo_search,
            "searx": self._searx_search,
            "brave": self._brave_search,
            "yandex": self._yandex_search
        }

    def _init_basic_vocab(self) -> None:
        """Initialize with ASCII characters and common symbols"""
        basic_chars = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ "
        for char in basic_chars:
            if char not in self.stoi:
                idx = len(self.stoi)
                self.stoi[char] = idx
                self.itos[idx] = char
        self.save_vocab()

    def load_vocab(self) -> None:
        """Load vocabulary from file"""
        self.stoi = self.special_tokens.copy()
        self.itos = {i: tok for tok, i in self.stoi.items()}
        
        if os.path.exists(self.vocab_file):
            try:
                with open(self.vocab_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for token, idx in data.items():
                        if token not in self.stoi and len(self.stoi) < MAX_VOCAB_SIZE:
                            self.stoi[token] = idx
                            self.itos[idx] = token
                logger.info(f"Loaded vocabulary with {len(self.stoi)} tokens")
            except Exception as e:
                logger.error(f"Vocabulary load error: {e}, starting fresh")

    def load_neural_memory(self) -> None:
        """Load neural memory patterns"""
        try:
            if os.path.exists(NEURAL_MEMORY_FILE):
                with open(NEURAL_MEMORY_FILE, 'r', encoding='utf-8') as f:
                    self.neural_memory = json.load(f)
            else:
                # Default neural patterns
                self.neural_memory = {
                    "programming": ["function", "class", "def", "import", "return", "algorithm"],
                    "science": ["quantum", "neural", "atom", "molecule", "biology", "physics"],
                    "medical": ["diagnosis", "treatment", "symptom", "patient", "disease", "therapy"],
                    "finance": ["stock", "market", "investment", "currency", "crypto", "blockchain"]
                }
        except Exception as e:
            logger.error(f"Neural memory load error: {e}")
            self.neural_memory = {}

    def load_quantum_sequences(self) -> None:
        """Load quantum sequences"""
        try:
            if os.path.exists(QUANTUM_SEQUENCES_FILE):
                with open(QUANTUM_SEQUENCES_FILE, 'r', encoding='utf-8') as f:
                    self.quantum_sequences = json.load(f)
            else:
                # Default quantum sequences
                self.quantum_sequences = {
                    "entanglement": "<|quantum|><|entangle|>",
                    "superposition": "<|quantum|><|superpose|>",
                    "interference": "<|quantum|><|interfere|>",
                    "teleportation": "<|quantum|><|teleport|>",
                    "decoherence": "<|quantum|><|decohere|>"
                }
        except Exception as e:
            logger.error(f"Quantum sequences load error: {e}")
            self.quantum_sequences = {}

    def save_vocab(self) -> None:
        """Save vocabulary to file"""
        with open(self.vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary saved with {len(self.stoi)} tokens")

    def save_neural_memory(self) -> None:
        """Save neural memory to file"""
        with open(NEURAL_MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.neural_memory, f, ensure_ascii=False, indent=2)

    def save_quantum_sequences(self) -> None:
        """Save quantum sequences to file"""
        with open(QUANTUM_SEQUENCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.quantum_sequences, f, ensure_ascii=False, indent=2)

    def normalize_text(self, text: str) -> str:
        """Advanced text normalization"""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove HTML tags
        text = self.patterns['html_tag'].sub('', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def neural_preprocess(self, text: str) -> str:
        """Advanced neural preprocessing with pattern recognition"""
        # Replace known patterns
        text = self.patterns['email'].sub('<|email|>', text)
        text = self.patterns['url'].sub('<|url|>', text)
        text = self.patterns['number'].sub('<|num|>', text)
        text = self.patterns['phone'].sub('<|phone|>', text)
        text = self.patterns['hashtag'].sub('<|entity|>', text)
        text = self.patterns['mention'].sub('<|entity|>', text)
        
        # Apply neural memory patterns
        for category, terms in self.neural_memory.items():
            for term in terms:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                text = pattern.sub(f'<|entity|>', text)
        
        return text

    def tokenize(self, text: str, 
                 add_special_tokens: bool = True, 
                 quantum_mode: bool = False,
                 max_length: int = None) -> List[int]:
        """
        Tokenize text with advanced options
        - quantum_mode: Enable quantum-inspired tokenization
        - max_length: Truncate tokens to specified length
        """
        # Normalization and neural preprocessing
        text = self.normalize_text(text)
        text = self.neural_preprocess(text)
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.stoi["<|start|>"])
        
        # Quantum-inspired encoding
        if quantum_mode:
            tokens.extend(self._quantum_encode(text))
        else:
            if self.mode == 'char':
                tokens.extend(self._char_tokenize(text))
            elif self.mode == 'word':
                tokens.extend(self._word_tokenize(text))
            elif self.mode == 'hybrid':
                tokens.extend(self._hybrid_tokenize(text))
        
        if add_special_tokens:
            tokens.append(self.stoi["<|end|>"])
        
        # Truncate if max_length specified
        if max_length is not None and len(tokens) > max_length:
            if add_special_tokens:
                tokens = tokens[:max_length-1] + [tokens[-1]]
            else:
                tokens = tokens[:max_length]
        
        return tokens

    def _char_tokenize(self, text: str) -> List[int]:
        """Character-level tokenization"""
        tokens = []
        for char in text:
            if char in self.stoi:
                tokens.append(self.stoi[char])
            else:
                self._handle_unknown(char, tokens)
        return tokens

    def _word_tokenize(self, text: str) -> List[int]:
        """Word-level tokenization"""
        tokens = []
        words = self.patterns['word'].findall(text)
        for word in words:
            if word in self.stoi:
                tokens.append(self.stoi[word])
            else:
                self._handle_unknown(word, tokens)
        return tokens

    def _hybrid_tokenize(self, text: str) -> List[int]:
        """Hybrid tokenization (words + special characters)"""
        tokens = []
        for match in self.patterns['word'].finditer(text):
            word = match.group()
            if word in self.stoi:
                tokens.append(self.stoi[word])
            else:
                self._handle_unknown(word, tokens)
        return tokens

    def _quantum_encode(self, text: str) -> List[int]:
        """Quantum-inspired encoding with entanglement"""
        tokens = []
        i = 0
        while i < len(text):
            # Check for quantum sequences
            found_sequence = False
            for seq in self.quantum_sequences.values():
                if text[i:].startswith(seq):
                    tokens.append(self.stoi[seq])
                    i += len(seq)
                    found_sequence = True
                    break
            
            if not found_sequence:
                # Check for special tokens
                if text[i:].startswith("<|") and "|>" in text[i:]:
                    end_pos = text.find("|>", i) + 2
                    token = text[i:end_pos]
                    if token in self.stoi:
                        tokens.append(self.stoi[token])
                        i = end_pos
                        continue
                
                # Process normally
                char = text[i]
                if char in self.stoi:
                    tokens.append(self.stoi[char])
                else:
                    self._handle_unknown(char, tokens)
                i += 1
        return tokens

    def _handle_unknown(self, token: str, tokens_list: list) -> None:
        """Handle unknown tokens with dynamic expansion"""
        if len(self.stoi) < MAX_VOCAB_SIZE:
            new_id = len(self.stoi)
            self.stoi[token] = new_id
            self.itos[new_id] = token
            tokens_list.append(new_id)
            # Log new token creation
            logger.debug(f"Created new token: {token} -> {new_id}")
        else:
            tokens_list.append(self.stoi["<|unk|>"])
            logger.warning(f"Vocabulary full, using <|unk|> for: {token}")

    def decode(self, tokens: List[int], 
              remove_special_tokens: bool = True, 
              quantum_mode: bool = False,
              skip_unknown: bool = False) -> str:
        """Decode tokens back to text"""
        text = []
        special_tokens = set(self.special_tokens.values())
        
        for token in tokens:
            if token in self.itos:
                token_str = self.itos[token]
                
                # Handle special tokens
                if remove_special_tokens and token in special_tokens:
                    continue
                
                # Handle quantum sequences
                if quantum_mode and token_str in self.quantum_sequences.values():
                    text.append(token_str)
                else:
                    text.append(token_str)
            else:
                if not skip_unknown:
                    text.append('<|unk|>')
        
        return ''.join(text)

    def update_from_text(self, text: str) -> set:
        """Update vocabulary from new text"""
        new_tokens = set()
        text = self.normalize_text(text)
        
        if self.mode == 'char':
            tokens = list(text)
        elif self.mode == 'word':
            tokens = self.patterns['word'].findall(text)
        else:  # hybrid
            tokens = [match.group() for match in self.patterns['word'].finditer(text)]
        
        for token in tokens:
            if token not in self.stoi and len(self.stoi) < MAX_VOCAB_SIZE:
                new_id = len(self.stoi)
                self.stoi[token] = new_id
                self.itos[new_id] = token
                new_tokens.add(token)
        
        if new_tokens:
            self.save_vocab()
            logger.info(f"Added {len(new_tokens)} new tokens to vocabulary")
        return new_tokens

    def train_on_corpus(self, corpus: List[str], batch_size: int = 1000) -> None:
        """Train tokenizer on a corpus of texts"""
        logger.info(f"Training on corpus with {len(corpus)} documents")
        new_tokens = set()
        
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i+batch_size]
            for text in batch:
                new_tokens.update(self.update_from_text(text))
        
        logger.info(f"Total new tokens added: {len(new_tokens)}")

    def add_neural_pattern(self, category: str, pattern: str) -> None:
        """Add a new neural memory pattern"""
        if category not in self.neural_memory:
            self.neural_memory[category] = []
        if pattern not in self.neural_memory[category]:
            self.neural_memory[category].append(pattern)
            self.save_neural_memory()
            logger.info(f"Added pattern to neural memory: {category} -> {pattern}")

    def add_quantum_sequence(self, name: str, sequence: str) -> None:
        """Add a new quantum sequence"""
        if name not in self.quantum_sequences:
            self.quantum_sequences[name] = sequence
            self.save_quantum_sequences()
            logger.info(f"Added quantum sequence: {name} -> {sequence}")

    def web_lookup(self, query: str, 
                  engine: str = "google", 
                  max_results: int = 3, 
                  cache: bool = True,
                  retries: int = 2) -> List[Dict[str, str]]:
        """
        🔍 Enhanced web search with multiple engines and caching
        Supported engines: google, bing, ddg (DuckDuckGo), searx, brave, yandex
        """
        # Create cache-safe filename
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"{engine}_{query_hash}.json"
        
        # Check cache first (with expiration)
        if cache and cache_file.exists():
            try:
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age < MAX_CACHE_AGE:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    logger.info("Cache expired, refreshing results")
            except Exception as e:
                logger.error(f"Cache read error: {e}")
        
        # Fetch from web with retries
        results = []
        if engine in self.search_engines:
            for attempt in range(retries + 1):
                try:
                    results = self.search_engines[engine](query, max_results)
                    if results:
                        break
                    logger.warning(f"Attempt {attempt+1} returned no results")
                except Exception as e:
                    logger.error(f"Search error (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
        else:
            logger.error(f"Unknown search engine: {engine}")
        
        # Save to cache if we got results
        if cache and results:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Cache write error: {e}")
        
        return results

    def _google_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Google search implementation with proper parsing"""
        try:
            url = f"https://www.google.com/search?q={quote(query)}&num={max_results+2}"
            response = requests.get(url, headers=self._get_headers(), timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.select("div.g"):
                # Skip sponsored results
                if result.find('div', class_='uEierd'):
                    continue
                
                title = result.select_one("h3")
                link = result.select_one("a")
                snippet = result.select_one(".VwiC3b, .MUxGbd")
                
                if title and link and snippet:
                    # Extract clean URL
                    parsed_url = urlparse(link['href'])
                    if parsed_url.path.startswith('/url?q='):
                        clean_url = parsed_url.query.split('&')[0][2:]
                    else:
                        clean_url = link['href']
                    
                    results.append({
                        'title': title.get_text(),
                        'url': clean_url,
                        'snippet': snippet.get_text()
                    })
                    if len(results) >= max_results:
                        break
            return results
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []

    def _bing_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Bing search implementation"""
        try:
            url = f"https://www.bing.com/search?q={quote(query)}&count={max_results}"
            response = requests.get(url, headers=self._get_headers(), timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.select("li.b_algo"):
                title = result.select_one("h2")
                link = result.select_one("a")
                snippet = result.select_one("p")
                
                if title and link and snippet:
                    results.append({
                        'title': title.get_text(),
                        'url': link['href'],
                        'snippet': snippet.get_text()
                    })
            return results[:max_results]
        except Exception as e:
            logger.error(f"Bing search error: {e}")
            return []

    def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """DuckDuckGo search implementation"""
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            response = requests.post(url, headers=self._get_headers(), timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.select(".result"):
                title = result.select_one(".result__a")
                link = result.select_one(".result__url")
                snippet = result.select_one(".result__snippet")
                
                if title and snippet:
                    # Extract actual URL from redirect
                    href = title.get('href', '')
                    if href.startswith('/'):
                        href = f"https://duckduckgo.com{href}"
                    
                    results.append({
                        'title': title.get_text(),
                        'url': href,
                        'snippet': snippet.get_text()
                    })
                    if len(results) >= max_results:
                        break
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def _searx_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Searx meta-search engine implementation"""
        try:
            # Use a public Searx instance
            url = f"https://searx.be/search?q={quote(query)}&format=json"
            response = requests.get(url, headers=self._get_headers(), timeout=20)
            data = response.json()
            
            results = []
            for result in data.get('results', [])[:max_results]:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'snippet': result.get('content', '')
                })
            return results
        except Exception as e:
            logger.error(f"Searx search error: {e}")
            return []
    
    def _brave_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Brave search implementation"""
        try:
            url = f"https://search.brave.com/search?q={quote(query)}"
            response = requests.get(url, headers=self._get_headers(), timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.select(".snippet"):
                title = result.select_one(".snippet-title")
                link = result.select_one(".snippet-title a")
                snippet = result.select_one(".snippet-description")
                
                if title and link and snippet:
                    results.append({
                        'title': title.get_text().strip(),
                        'url': link['href'],
                        'snippet': snippet.get_text().strip()
                    })
                    if len(results) >= max_results:
                        break
            return results
        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return []
    
    def _yandex_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Yandex search implementation"""
        try:
            url = f"https://yandex.com/search/?text={quote(query)}"
            response = requests.get(url, headers=self._get_headers(), timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.select(".serp-item"):
                title = result.select_one(".organic__title")
                link = result.select_one(".path a")
                snippet = result.select_one(".organic__content-wrapper")
                
                if title and link and snippet:
                    results.append({
                        'title': title.get_text().strip(),
                        'url': link['href'],
                        'snippet': snippet.get_text().strip()
                    })
                    if len(results) >= max_results:
                        break
            return results
        except Exception as e:
            logger.error(f"Yandex search error: {e}")
            return []

    def get_contextual_answer(self, query: str, 
                             engine: str = "google", 
                             max_context: int = 500) -> str:
        """Get contextual answer from web search"""
        results = self.web_lookup(query, engine, max_results=3)
        context = ""
        
        for result in results:
            context += f"{result['title']}: {result['snippet']}\n"
            if len(context) > max_context:
                break
        
        return context.strip()

    def _get_headers(self) -> Dict[str, str]:
        """Generate request headers with random user agent"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
        ]
        
        return {
            "User-Agent": random.choice(user_agents),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Accept-Encoding": "gzip, deflate, br"
        }

    def vocab_size(self) -> int:
        return len(self.stoi)

    def __call__(self, text: str, **kwargs) -> List[int]:
        return self.tokenize(text, **kwargs)

    def batch_encode(self, texts: List[str], **kwargs) -> List[List[int]]:
        return [self.tokenize(text, **kwargs) for text in texts]

    def batch_decode(self, batch_tokens: List[List[int]], **kwargs) -> List[str]:
        return [self.decode(tokens, **kwargs) for tokens in batch_tokens]
    
    def get_state(self) -> Dict:
        """
        Serializes the tokenizer's state for saving
        Includes all core internal mappings and configurations
        """
        return {
            'mode': self.mode,
            'special_tokens': self.special_tokens,
            'stoi': self.stoi,
            'itos': self.itos,
            'neural_memory': self.neural_memory,
            'quantum_sequences': self.quantum_sequences
        }

    def set_state(self, state_dict: Dict) -> None:
        """
        Alias for load_state for backward compatibility
        Restores tokenizer state from a state dictionary
        """
        self.load_state(state_dict)

    def load_state(self, state_dict: Dict) -> None:
        """
        Loads a previously saved tokenizer state
        """
        self.mode = state_dict.get('mode', 'hybrid')
        self.special_tokens = state_dict.get('special_tokens', self._init_special_tokens())
        self.stoi = state_dict.get('stoi', {})
        self.itos = state_dict.get('itos', {})
        self.neural_memory = state_dict.get('neural_memory', {})
        self.quantum_sequences = state_dict.get('quantum_sequences', {})
        
        # Reinitialize patterns and search engines
        self.patterns = self._init_patterns()
        self.search_engines = self._init_search_engines()
        
        logger.info("Tokenizer state loaded successfully")

    def save_full_state(self, file_path: str) -> None:
        """Save complete tokenizer state to a file"""
        state = self.get_state()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(f"Full tokenizer state saved to {file_path}")

    def load_full_state(self, file_path: str) -> None:
        """Load complete tokenizer state from a file"""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.load_state(state)
        else:
            logger.error(f"State file not found: {file_path}")

    def clear_cache(self) -> None:
        """Clear all cached search results"""
        for file in self.cache_dir.glob("*.json"):
            try:
                file.unlink()
            except Exception as e:
                logger.error(f"Error deleting {file}: {e}")
        logger.info("Search cache cleared")

    def get_token(self, text: str, add_if_missing: bool = True) -> int:
        """
        Get token ID for a string
        - add_if_missing: Add to vocabulary if not present
        """
        if text in self.stoi:
            return self.stoi[text]
        elif add_if_missing and len(self.stoi) < MAX_VOCAB_SIZE:
            new_id = len(self.stoi)
            self.stoi[text] = new_id
            self.itos[new_id] = text
            return new_id
        else:
            return self.stoi["<|unk|>"]

    def get_token_text(self, token_id: int) -> str:
        """Get text for a token ID"""
        return self.itos.get(token_id, "<|unk|>")

    def analyze_text(self, text: str) -> Dict:
        """Analyze text and return token statistics"""
        tokens = self.tokenize(text)
        return {
            "char_count": len(text),
            "token_count": len(tokens),
            "unique_tokens": len(set(tokens)),
            "unknown_tokens": tokens.count(self.stoi["<|unk|>"]),
            "compression_ratio": len(text) / len(tokens) if tokens else 0
        }

    def merge_vocabularies(self, other_tokenizer: 'NeuroQuantumTokenizer') -> None:
        """
        Merge vocabulary from another tokenizer instance
        Useful for distributed training scenarios
        """
        new_tokens = 0
        for token, idx in other_tokenizer.stoi.items():
            if token not in self.stoi and len(self.stoi) < MAX_VOCAB_SIZE:
                new_id = len(self.stoi)
                self.stoi[token] = new_id
                self.itos[new_id] = token
                new_tokens += 1
        
        if new_tokens:
            self.save_vocab()
            logger.info(f"Merged {new_tokens} tokens from another tokenizer")

    def optimize_vocabulary(self, min_frequency: int = 2) -> None:
        """
        Optimize vocabulary by removing infrequent tokens
        - min_frequency: Minimum occurrence count to keep token
        """
        # This would require tracking token frequencies (not implemented)
        logger.warning("optimize_vocabulary requires frequency tracking not implemented")
        # Implementation would require maintaining frequency counts
        # and rebuilding stoi/itos after removing low-frequency tokens

# tokenizer/tokenizer.py (recommended addition)
@property
def vocab_size(self):
    return len(self.stoi)
