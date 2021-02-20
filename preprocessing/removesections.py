import re

def remove_sections(doc):
    if doc["structured"]:
        sections_whitelist = ["RESULTS", "CONCLUSIONS", "METHODS AND FINDINGS", "FINDINGS", "CONCLUSION"]
        section_regexp = r"^\b[A-Z]+[A-Z]+(?:\s*[,]*\s+[A-Z]+[A-Z]+)*\b"
        abstract = doc["abstract"]
        cur_section = ""

        with open("../datasets/cleaned_sections.txt") as f:
            cleaned_sections = sorted(f.read().splitlines(), key=len, reverse=True)

        for sentence_idx, sentence in enumerate(abstract):
            # Find and clean new section name
            if re.search(section_regexp, sentence):
                for cleaned_section in cleaned_sections:
                    if cleaned_section in re.findall(section_regexp, sentence)[0]:
                        cur_section = cleaned_section
                        break
            
            # Rewrite sentence in non-whitelisted section
            if cur_section not in sections_whitelist:
                abstract[sentence_idx] = ""
        
        doc["abstract"] = abstract
    return doc
    


            
