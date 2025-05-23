import concurrent.futures
import os
import re
import json
import dspy
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import as_completed
from typing import Union, List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.ArticleTextProcessing import ArticleTextProcessing


script_dir = os.path.dirname(os.path.abspath(__file__))


class ConceptGenerator(dspy.Module):
    """Extract information and generate a list of concepts."""
    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.lm = lm
        self.concept_generator = dspy.Predict(GenConcept)

    def forward(self, infos: List[Dict]):
        snippets_list = []
        for info in infos:
            snippet = info.get('snippets', [])
            snippets_list.extend(snippet)
        
        snippets_list_str = "\n".join(f"{index + 1}. {snippet}" for index, snippet in enumerate(snippets_list))
        snippets_list_str = ArticleTextProcessing.limit_word_count_preserve_newline(snippets_list_str, 3000)

        with dspy.settings.context(lm=self.lm):
            concepts = self.concept_generator(info=snippets_list_str).concepts

        pattern = r"\d+\.\s*(.*)"
        matches = re.findall(pattern, concepts)
        concept_list = [match.strip() for match in matches]

        return concept_list

class ExtendConcept(dspy.Signature):
    """You are an analytical robot. I will provide you with a subject, the information I have searched about it, and our preliminary concept of it. I need you to generate a detailed, in-depth, and insightful report based on it, further exploring our initial ideas. 

First, break down the subject into several broad categories, then create corresponding search engine keywords for each category. 

Note: The new categories should not repeat the previous ones. 

Your output format should be as follows:  
-[Category 1]  
--{Keyword 1}  
--{Keyword 2}  
-[Category 2]  
--{Keyword 1}  
--{Keyword 2}"""
    info = dspy.InputField(prefix='The information you have collected from the webpage:', format=str)
    concept = dspy.InputField(prefix='The summary of the previous concepts:', format=str)
    category = dspy.InputField(prefix='The broader categories you need to further expand:', format=str)
    keywords = dspy.OutputField(format=str)


class GenConcept(dspy.Signature):
    """Please analyze, summarize, and evaluate the following webpage information. 
Think like a person, distill the core point of each piece of information, and synthesize them into a comprehensive opinion. 
Present your comprehensive opinion in the format of 1. 2. ..."""
    info = dspy.InputField(prefix='The webpage information you have collected:', format=str)
    concepts = dspy.OutputField(format=str)


class MindPoint():
    def __init__(self, retriever, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel], root: bool = False,
                 children: Optional[List['MindPoint']] = None, concept: str = '',
                 info: Optional[List[Dict]] = None, category: str = ''):
        self.root = root
        self.category = category
        self.children = children if children is not None else {}
        self.concept = concept
        self.info = info if info is not None else []
        self.lm = lm
        self.retriever = retriever
        self.concept_generator = ConceptGenerator(lm=lm)
    
    def extend(self):
        extend_concept = dspy.Predict(ExtendConcept)
        with dspy.settings.context(lm=self.lm):
            info='\n'.join([str(i) for i in self.info])
            keywords = extend_concept(info='\n'.join([str(i) for i in self.info]), concept=self.concept, category = self.category).keywords
        categories = {}
        current_category = None
        for line in keywords.split('\n'):
            line = line.strip()
            if (line.startswith('-[') and line.endswith(']')) or (line.startswith('- [') and line.endswith(']')):
                current_category = line[2:-1]
                categories[current_category] = []
            elif (line.startswith('--{') and current_category) or (line.startswith('-- {') and current_category):
                keyword = line[3:-1].strip()
                if keyword:
                    categories[current_category].append(keyword)

        for category, keywords_list in categories.items():
            new_info = self.retriever(keywords_list)
            new_concept = self.concept_generator.forward(new_info)
            new_node = MindPoint(concept=new_concept, info=new_info, lm=self.lm, retriever=self.retriever, category=category)
            self.children[category] = new_node


class MindMap():
    def __init__(self, 
                 retriever, 
                 gen_concept_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 depth: int,
                 workers: int = 5
                 ):
        self.retriever = retriever
        self.gen_concept_lm = gen_concept_lm
        self.depth = depth
        self.concept_generator = ConceptGenerator(lm=self.gen_concept_lm)
        self.root = None
        self.max_workers = workers
        print('MindMap initialized')

    def build_map(self, topic: str):
        root_info = self.retriever(topic)
        root_concept = self.concept_generator(root_info)
        root = MindPoint(root=True, info=root_info, concept=root_concept, lm=self.gen_concept_lm, retriever=self.retriever, category=topic)
        self.root = root
        
        current_level = [root]
        
        for count in range(self.depth):
            next_level = []

            yield current_level
            if count == self.depth - 1:  # Check if it's the last layer
                break
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(node.extend): node for node in current_level}
                
                for future in concurrent.futures.as_completed(futures):
                    node = futures[future]
                    # Assuming extend populates children.
                    next_level.extend(node.children.values())
            
            yield current_level
            current_level = next_level
    
    def recursive_extend(self, node: MindPoint, count: int):
        if count >= self.depth:
            return
        node.extend()
        count += 1


    def save_map(self, root: MindPoint, filename: str):
        def serialize_node(node: MindPoint):
            return {
                'category': node.category,
                'concept': node.concept,
                'children': {k: serialize_node(v) for k, v in node.children.items()},
                'info':node.info,
            }
        
        mind_map_dict = serialize_node(root)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(mind_map_dict, f, ensure_ascii=False, indent=2)

    def load_map(self, filename: str):
        def deserialize_node(node_data):
            category = node_data['category']
            concept = node_data['concept']
            info = node_data['info']
            children_data = node_data['children']

            node = MindPoint(concept=concept, info=info, lm=self.gen_concept_lm, retriever=self.retriever, category=category)
            node.children = {k: deserialize_node(v) for k, v in children_data.items()}
            return node
        
        with open(filename, 'r', encoding='utf-8') as f:
            mind_map_dict = json.load(f)
        
        self.root = deserialize_node(mind_map_dict)
        return self.root

    def export_categories_and_concepts(self) -> str:
        root = self.root
        output = []

        def traverse(node: MindPoint, indent=0):
            output.append(" " * indent + node.category)
            for concept in node.concept:
                output.append(" " * (indent + 2) + concept)
            for child in node.children.values():
                traverse(child, indent + 2)

        traverse(root)
        return "\n".join(output)

    def get_all_infos(self) -> List[Dict[str, any]]:
        """
        Get all unique info from the MindMap, ensuring unique URLs.
        """
        all_infos = []
        seen_urls = set()

        def traverse(node: MindPoint):
            if node.info:
                for info in node.info:
                    url = info.get('url')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_infos.append(info)
            for child in node.children.values():
                traverse(child)

        traverse(self.root)
        self.all_infos = all_infos
        return all_infos

    def prepare_table_for_retrieval(self):
        """
        Prepare collected snippets and URLs for retrieval by encoding the snippets using paraphrase-MiniLM-L6-v2.
        collected_urls and collected_snippets have corresponding indices.
        """
        # self.encoder = SentenceTransformer('/mnt/nas-alinlp/xizekun/huggingface_cache/all-MiniLM-L6-v2')
        self.encoder = SentenceTransformer('/tmp/pycharm_project_773/Models')
        self.collected_urls = []
        self.collected_snippets = []
        seen_urls = set()

        for info in self.get_all_infos():
            url = info.get('url')
            snippets = info.get('snippets', [])
            if url and url not in seen_urls:
                seen_urls.add(url)
                for snippet in snippets:
                    self.collected_urls.append(url)
                    self.collected_snippets.append(snippet)

        self.encoded_snippets = self.encoder.encode(self.collected_snippets, show_progress_bar=True)

    def retrieve_information(self, queries: Union[List[str], str], search_top_k) -> List[Dict[str, any]]:
        """
        Retrieve relevant information based on the given queries.
        Returns a list of dictionaries containing 'url' and 'snippets'.
        """
        selected_urls = []
        selected_snippets = []
        if type(queries) is str:
            queries = [queries]
        for query in queries:
            encoded_query = self.encoder.encode(query, show_progress_bar=False)
            sim = cosine_similarity([encoded_query], self.encoded_snippets)[0]
            sorted_indices = np.argsort(sim)
            for i in sorted_indices[-search_top_k:][::-1]:
                selected_urls.append(self.collected_urls[i])
                selected_snippets.append(self.collected_snippets[i])

        url_to_snippets = {}
        for url, snippet in zip(selected_urls, selected_snippets):
            if url not in url_to_snippets:
                url_to_snippets[url] = set()
            url_to_snippets[url].add(snippet)

        result = []
        for url, snippets in url_to_snippets.items():
            result.append({
                'url': url,
                'snippets': list(snippets)
            })

        return result

    def visualize_map(self, root: MindPoint):
        G = nx.DiGraph()

        def add_edges(node: MindPoint, parent=None):
            if parent is not None:
                G.add_edge(parent, node.category)
            for child in node.children.values():
                add_edges(child, node.category)
        
        add_edges(root)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
        plt.title("MindMap Visualization", fontsize=15)
        plt.show()
