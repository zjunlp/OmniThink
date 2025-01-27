import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union
import random
import dspy
import sys

from src.utils.ArticleTextProcessing import ArticleTextProcessing

# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]
class ArticleGenerationModule():
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage, 
    """

    def __init__(self,
                 retriever,
                 article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 retrieve_top_k: int = 10,
                 max_thread_num: int = 10,
                ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.retriever = retriever
        self.section_gen = ConvToSection(engine=self.article_gen_lm)
        self.leaf_section_gen = ConvToLeafSection(engine=self.article_gen_lm)
        self.nonleaf_section_gen = ConvToNonLeafSection(engine=self.article_gen_lm)
        self.summarize_leaf_section = SumToNonLeafSection(engine=self.article_gen_lm)

    def generate_section(self, topic, section_name, mindmap, section_query, section_outline):
        collected_info = mindmap.retrieve_information(queries=section_query,
                                                                    search_top_k=self.retrieve_top_k)
        output = self.section_gen(
            topic=topic,
            outline=section_outline,
            section=section_name,
            collected_info=collected_info,
        )

        return {"section_name": section_name, "section_content": output.section, "collected_info": collected_info}

    def generate_nonleaf_section(self,section, topic,  mindmap):
        
        children_content = []
        for child in section.children:
            children_content.append(child.content)
        
        children_content = "\n".join(children_content)

        output = self.nonleaf_section_gen(
            topic=topic,
            section=section.section_name,
            contents=children_content,
        )

        return {"section_name": section.section_name, "section_content": output.section, "collected_info": children_content}
        
    def sum_leaf_section(self,section, topic,  mindmap):
        
        children_content = []
        for child in section.children:
            children_content.append(child.content)
        
        children_content = "\n".join(children_content)

        output = self.summarize_leaf_section(
            topic=topic,
            section=section.section_name,
            contents=children_content,
        )

        return {"section_name": section.section_name, "section_content": output.section, "collected_info": children_content}
   

    def generate_left_section(self, topic, section_name, mindmap):
        collected_info = mindmap.retrieve_information(queries=section_name,
                                                                    search_top_k=self.retrieve_top_k)
        output = self.leaf_section_gen(
            topic=topic,
            outline=None,
            section=section_name,
            collected_info=collected_info,
        )

        return {"section_name": section_name, "section_content": output.section, "collected_info": collected_info}

    

    def generate_article(self,
                         topic: str,
                         mindmap,
                         article_with_outline,
                         ):
        """
        Generate article for the topic based on the information table and article outline.
        """
        mindmap.prepare_table_for_retrieval()

        sections_to_write = article_with_outline.get_first_level_section_names()
        section_output_dict_collection = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            future_to_sec_title = {}
            for section_title in sections_to_write:
                section_query = article_with_outline.get_outline_as_list(
                    root_section_name=section_title, add_hashtags=False
                )
                queries_with_hashtags = article_with_outline.get_outline_as_list(
                    root_section_name=section_title, add_hashtags=True
                )
                section_outline = "\n".join(queries_with_hashtags)

                future_to_sec_title[
                    executor.submit(self.generate_section, 
                                    topic, section_title, mindmap, section_query,section_outline)
                ] = section_title

            for future in concurrent.futures.as_completed(future_to_sec_title):
                section_output_dict_collection.append(future.result())

        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(parent_section_name=topic,
                                   current_section_content=section_output_dict["section_content"],
                                   current_section_info_list=section_output_dict["collected_info"],
                                )

        article.post_processing()

        return article

    def generate_deep_article(self,
                         topic: str,
                         mindmap,
                         article_with_outline,
                         ):
        """
        Generate article by depth-first traversal.
        """
        def deepest_first_traversal(root):
            """
            多叉树深度优先遍历：从最深叶节点到根节点
            """
            # 记录每个节点的深度
            depth_map = {}
            stack = [(root, 0)]  # (节点, 深度)

            # 迭代DFS遍历整棵树
            while stack:
                node, depth = stack.pop()
                depth_map[node] = depth  # 记录节点深度
                
                # 将子节点逆序压栈（保证DFS顺序正确）
                for child in reversed(node.children):
                    stack.append((child, depth + 1))

            # 按深度降序排序节点
            sorted_nodes = sorted(depth_map.items(), key=lambda x: -x[1])
            
            # 提取节点
            return [node for node, _ in sorted_nodes]

        def all_children_leaf(root):
            
            children = root.children
            if len(children)==0:
                return False
            for child in children:
                if len( child.children ) >0:
                    return False 
            return True

        mindmap.prepare_table_for_retrieval()

        sections = deepest_first_traversal(article_with_outline.root)

        for sec in sections:
            if len(sec.children) == 0:
                ret = self.generate_left_section(topic, sec.section_name, mindmap)
            elif all_children_leaf(sec):
                ret = self.sum_leaf_section(sec, topic, mindmap)
            else:
                ret = self.generate_nonleaf_section(sec, topic, mindmap)
            sec.content = ret["section_content"]

        # clean leaf node content
        for sec in sections:
            if len(sec.children) == 0:
                sec.content = None


        article_with_outline.post_processing()

        return article_with_outline

class ConvToSection(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def forward(self, topic: str, outline:str, section: str, collected_info: List):
        all_info = ''
        for idx, info in enumerate(collected_info):
            all_info += f'[{idx + 1}]\n' + '\n'.join(info['snippets'])
            all_info += '\n\n'

        all_info = ArticleTextProcessing.limit_word_count_preserve_newline(all_info, 1500)

        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(topic=topic, info=info, section=section).output)
         
        section = section.replace('\[','[').replace('\]',']')
        return dspy.Prediction(section=section)

class ConvToLeafSection(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteLeafSection)
        self.engine = engine

    def forward(self, topic: str, outline:str, section: str, collected_info: List):
        all_info = ''
        for idx, info in enumerate(collected_info):
            all_info += f'[{idx + 1}]\n' + '\n'.join(info['snippets'])
            all_info += '\n\n'

        # all_info = ArticleTextProcessing.limit_word_count_preserve_newline(all_info, 1500)

        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(topic=topic, info=info, section=section).output)
         
        section = section.replace('\[','[').replace('\]',']')
        return dspy.Prediction(section=section)

class ConvToNonLeafSection(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteNonLeafSection)
        self.engine = engine

    def forward(self, topic: str, section: str, contents:str):
        # all_info = ArticleTextProcessing.limit_word_count_preserve_newline(all_info, 1500)

        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(subsections=contents, topic=topic, section=section).output)
         
        section = section.replace('\[','[').replace('\]',']')
        return dspy.Prediction(section=section)

class SumToNonLeafSection(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(SummarizeLeafSection)
        self.engine = engine

    def forward(self, topic: str, section: str, contents:str):
        # all_info = ArticleTextProcessing.limit_word_count_preserve_newline(all_info, 1500)

        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(subsections=contents, topic=topic, section=section).output)
         
        section = section.replace('\[','[').replace('\]',']')
        return dspy.Prediction(section=section)

class WriteSection(dspy.Signature):
    """Write a Wikipedia section based on the collected information.

    Here is the format of your writing:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
        3. The language style should resemble that of Wikipedia: concise yet informative, formal yet accessible.
    """
    info = dspy.InputField(prefix="The Collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str)


class WriteLeafSection(dspy.Signature):
    """Write a Wikipedia section based on the collected information.

    Here is the format of your writing:
        1. Include as much information as possible.
        2. Include as much quantitative data as possible.
        3. Don't stray from the topic. 
        4. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
        5. The language style should resemble that of Wikipedia: concise yet informative, formal yet accessible.
    """
    info = dspy.InputField(prefix="The Collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str)

class SummarizeLeafSection(dspy.Signature):
    """Summarize the contents of the subsection and include as much information as possible, especially quantitative data.
    Here is the format of your writing:
        1. Include as much information as possible.
        2. Include as much quantitative data as possible.
        3. Don't stray from the topic. 
        4. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
        5. The language style should resemble that of Wikipedia: concise yet informative, formal yet accessible.
    """

    subsections = dspy.InputField(prefix="Content of subsections:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str)


class WriteNonLeafSection(dspy.Signature):
    """Use the most concise language possible to introduce the contents of the subsections.
    Here is the format of your writing:
    1. Don't contain too much detail.
    2. This is an introductory chapter and does not need to contain all the contents.
    3. Prefer to lose information rather than duplicate content in subsections.
    """

    subsections = dspy.InputField(prefix="Content of subsections:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str)