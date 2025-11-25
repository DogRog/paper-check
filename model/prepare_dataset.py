import json
import os
import re
from tqdm import tqdm

# --- Configuration ---
# Ensure this path is correct relative to where you run the script
PEERREAD_PATH = "./PeerRead/data" 
OUTPUT_FILE = "peerread_qwen_fulltext.jsonl"
TARGET_CONFERENCES = ["iclr_2017", "nips_2017", "acl_2017"] 

# --- 1. Text Extraction Logic ---
def extract_sections(sections_list):
    """
    Extracts Introduction and Methods. 
    Includes safety checks for None types in headings.
    """
    full_text = ""
    target_headers = ["introduction", "method", "model", "approach", "experiment", "result"]
    
    for section in sections_list:
        if not isinstance(section, dict): continue
        
        # Safety: heading might be None
        raw_heading = section.get('heading')
        heading = (raw_heading if raw_heading else "").lower()
        text = section.get('text', '')
        
        # Skip if text is empty or it's a reference section
        if not text or "reference" in heading or "bibliography" in heading:
            continue
            
        # Heuristic: Keep if target header OR if we are still building the first 1500 chars (Abstract/Intro)
        if any(t in heading for t in target_headers) or len(full_text) < 1500:
            display_heading = raw_heading if raw_heading else "Section"
            full_text += f"\n\n## {display_heading}\n{text}"
    
    return full_text.strip()

def normalize_title(title):
    if not title: return ""
    return re.sub(r'[^a-z0-9]', '', title.lower())

# --- 2. The Local Loader ---
def load_local_peerread(base_path, conferences):
    data_pairs = []
    print(f"Scanning local data at: {os.path.abspath(base_path)}")
    
    for conf in conferences:
        for split in ["train", "dev", "test"]:
            folder_path = os.path.join(base_path, conf, split)
            
            # Paths for "Monolithic" JSONs (standard in some years)
            reviews_path = os.path.join(folder_path, "reviews.json")
            pdfs_path = os.path.join(folder_path, "parsed_pdfs.json")
            
            # Paths for "Directory" style (standard in others)
            reviews_dir = os.path.join(folder_path, "reviews")
            pdfs_dir = os.path.join(folder_path, "parsed_pdfs")
            
            # Check existence
            if not (os.path.exists(reviews_path) or os.path.isdir(reviews_dir)):
                # Only warn if it's strictly missing (some conferences lack 'dev' sets)
                if split == "train":
                    print(f"  [WARN] Skipping {conf}/{split}: No data found.")
                continue
                
            print(f"Processing {conf} [{split}]...")
            
            # --- A. Load Reviews ---
            papers_data = []
            if os.path.exists(reviews_path):
                # Load Monolithic
                with open(reviews_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    papers_data = raw.get('reviews', []) if isinstance(raw, dict) else raw
            elif os.path.isdir(reviews_dir):
                # Load Directory
                for fn in sorted(os.listdir(reviews_dir)):
                    if not fn.lower().endswith(".json"): continue
                    try:
                        with open(os.path.join(reviews_dir, fn), 'r', encoding='utf-8') as f:
                            raw = json.load(f)
                            if isinstance(raw, dict): papers_data.append(raw)
                            elif isinstance(raw, list): papers_data.extend(raw)
                    except json.JSONDecodeError:
                        continue

            # --- B. Load PDFs (Lookup Table) ---
            pdf_lookup = {}
            
            # Helper to add to lookup
            def add_to_lookup(title, sections):
                t = normalize_title(title)
                if t: pdf_lookup[t] = sections

            if os.path.exists(pdfs_path):
                # Load Monolithic
                with open(pdfs_path, 'r', encoding='utf-8') as f:
                    pdfs_data = json.load(f)
                    # Handle structure where data is inside 'paper_sections' or root list
                    sections_list = pdfs_data.get('paper_sections', []) if isinstance(pdfs_data, dict) else pdfs_data
                    for p in sections_list:
                        add_to_lookup(p.get('title'), p.get('sections', []))
                        
            elif os.path.isdir(pdfs_dir):
                # Load Directory
                for fn in sorted(os.listdir(pdfs_dir)):
                    if not fn.lower().endswith(".json"): continue
                    with open(os.path.join(pdfs_dir, fn), 'r', encoding='utf-8') as f:
                        p = json.load(f)
                        # PeerRead metadata structure varies wildly
                        title = p.get('title')
                        if not title and 'metadata' in p:
                            title = p['metadata'].get('title')
                        
                        sections = p.get('sections', [])
                        if not sections and 'metadata' in p:
                            sections = p['metadata'].get('sections', [])
                            
                        add_to_lookup(title, sections)

            # --- C. Merge ---
            count_for_split = 0
            for item in papers_data:
                title = item.get('title', '')
                norm_title = normalize_title(title)
                
                # 1. Get Full Text (or fallback to abstract)
                sections = pdf_lookup.get(norm_title)
                if sections:
                    paper_content = extract_sections(sections)
                else:
                    paper_content = f"Abstract: {item.get('abstract', '')}"
                
                # Filter: Too short? (Garbage data)
                if len(paper_content) < 200: continue
                
                # 2. Get Reviews
                # Handle inconsistent field names
                review_entries = item.get('reviews') or item.get('comments') or []
                if isinstance(review_entries, dict): review_entries = [review_entries]
                
                decision = "Accept" if item.get('accepted') else "Reject"
                
                for rev in review_entries:
                    # Handle raw string comments vs dict comments
                    if isinstance(rev, dict):
                        review_text = rev.get('comments', '')
                        score = rev.get('rating', 'N/A')
                    else:
                        review_text = str(rev)
                        score = "N/A"

                    # Fix missing scores based on decision
                    if score in ["N/A", None]:
                        if decision == "Accept":
                            score = 8 
                        else:
                            score = 3

                    if len(review_text) < 50: continue
                    
                    data_pairs.append({
                        "title": title,
                        "content": paper_content,
                        "review": review_text,
                        "decision": decision,
                        "score": score
                    })
                    count_for_split += 1
            
            print(f"  -> Merged {count_for_split} pairs.")

    return data_pairs

# --- 3. Formatter & Writer ---
def main():
    if not os.path.exists(PEERREAD_PATH):
        print(f"CRITICAL ERROR: Could not find {PEERREAD_PATH}")
        print("1. Go to your terminal.")
        print("2. Run: git clone https://github.com/allenai/PeerRead.git")
        print("3. Run: cd PeerRead && bash setup.sh")
        return

    # Load Data
    raw_data = load_local_peerread(PEERREAD_PATH, TARGET_CONFERENCES)
    
    if len(raw_data) == 0:
        print("ERROR: No data found. Did you run 'bash setup.sh' inside the PeerRead folder?")
        print("The repository only contains code; 'setup.sh' downloads the actual JSONs.")
        return

    print(f"Found {len(raw_data)} valid review pairs. Formatting for Qwen...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(raw_data):
            system_msg = (
                "You are an expert Area Chair for a top-tier AI conference. "
                "Critique the following paper based on: Novelty, Soundness, and Significance."
            )
            
            # Format: Structured Input
            user_msg = f"""
            TITLE: {item['title']}
            
            CONTENT:
            {item['content']}
            
            ---
            Task: Provide a detailed peer review and a final decision.
            """
            
            # Format: Structured Output (CoT)
            assistant_msg = f"""
            {item['review']}
            
            Decision: {item['decision']}
            Score: {item['score']}
            """
            
            json_line = {
                "conversations": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]
            }
            f.write(json.dumps(json_line) + "\n")
            
    print(f"Success! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()