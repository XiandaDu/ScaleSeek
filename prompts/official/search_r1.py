"""Search-R1 inference constants -- VERBATIM from PeterGriffinJin/Search-R1 `infer.py`.

Pinned commit: configs/methods/search_r1_*.yaml `official_repo.commit`.  Each constant
below is compared byte for byte with the same assignment in the official file
(scaleseek.official.OFFICIAL_SOURCES) before every run.  The official f-string's only
placeholder is {question}, so it is reproduced as a str.format template.

Also reproduced in scaleseek/methods/search_r1.py (code, not prompts): the question is
stripped and gets a trailing '?' if it lacks one, exactly as infer.py does.
"""

PROMPT_TEMPLATE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

SEARCH_TEMPLATE = '\n\n{output_text}<information>{search_results}</information>\n\n'

STOP_SEQUENCES = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]

EOS_TOKEN_IDS = [151645, 151643] # for Qwen2.5 series models


def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


def _passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
                    
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference
