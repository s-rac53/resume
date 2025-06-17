import os
import re
import numpy as np
import hashlib
import json
from PIL import Image
import pdf2image
from paddleocr import PaddleOCR
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio
import nest_asyncio
from typing import Union, List, Dict
import logging
import time
# Removed unused imports
# from datetime import datetime, timezone, timedelta
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
import urllib.parse
from tika import parser
from google.colab import userdata

# Import google-search
try:
    from googlesearch import search
except ImportError:
    print("google-search not installed. Run: pip install google-search-python")
    exit()

# === Setup Logging and Async ===
logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
nest_asyncio.apply()

# === Initialize OCR and Models ===
ocr = PaddleOCR(use_angle_cls=True, lang='en')
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-distilroberta-v1")

# Check if model file exists
model_path = "/content/drive/MyDrive/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
if not os.path.exists(model_path):
    logging.error(f"Model file not found: {model_path}")
    raise FileNotFoundError(f"Model file not found: {model_path}")
llm = LlamaCPP(model_path=model_path)

# === Cache for About Us Results ===
aboutus_cache = {}



# === Web Scraping Functions ===
def get_random_headers():
    """Return randomized User-Agent headers for page content fetching."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com"
    }

def clean_search_query(org_name: str, certificate_type: str = None) -> str:
    """Clean and normalize organization name for search with context-specific enhancements."""
    org_name = org_name.strip().upper()
    org_name = re.sub(r"^(clean name:\s*|the\s+|a\s+|an\s+)", "", org_name, flags=re.IGNORECASE)
    org_name = re.sub(r"TECHINOLOGICAL", "TECHNOLOGICAL", org_name)
    org_name = re.sub(r"UNVIERSITY", "UNIVERSITY", org_name)
    org_name = re.sub(r"INSTTUTE", "INSTITUTE", org_name)
    org_name = re.sub(r"\s+", " ", org_name)
    org_name = re.sub(r"[^\w\s]", " ", org_name)
    # Add context-specific suffixes for better search relevance
    if certificate_type in ["10th", "12th"]:
        org_name += " EDUCATION BOARD"
    elif certificate_type in ["bachelor", "master"]:
        org_name += " UNIVERSITY"
    elif certificate_type in ["work", "internship"]:
        org_name += " COMPANY"
    return org_name.strip()

def google_search_with_retry(query: str, num_results: int = 10, retries: int = 3, delay: int = 5) -> List[Dict]:
    """Perform Google search with retries and multiple query variations."""
    results = []
    query_variations = [
        query,
        f"{query} official website",
        f"{query} about us",
        f"site:*.edu.in {query}" if "university" in query.lower() or "board" in query.lower() else f"site:*.com {query}"
    ]

    for q in query_variations:
        for attempt in range(retries):
            try:
                for url in search(q, num_results=num_results):
                    results.append({"href": url, "title": ""})
                    if len(results) >= num_results:
                        break
                logging.info(f"Search succeeded for query: {q}, attempt {attempt + 1}")
                return results
            except Exception as e:
                logging.warning(f"Search attempt {attempt + 1} failed for {q}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
    logging.error(f"All {retries} search attempts failed for {query}")
    return []

def search_top_urls(query: str, max_results: int = 3) -> List[Dict]:
    """Scrape top URLs, prioritizing About Us or homepage links."""
    try:
        results = google_search_with_retry(query, num_results=10)
        exclude_extensions = [".pdf", ".doc", ".docx"]
        exclude_domains = ["linkedin.com", "facebook.com", "twitter.com", "instagram.com", "glassdoor.com", "wikipedia.org", "youtube.com"]
        filtered_results = []

        # Prioritize "About Us" or homepage URLs
        for result in results:
            href = result.get("href", "")
            title = result.get("title", "").lower()
            if (href and not any(href.lower().endswith(ext) for ext in exclude_extensions) and
                not any(x in href.lower() for x in exclude_domains)):
                if "about" in href.lower() or "about" in title or "/" == href.rstrip("/")[-1]:
                    filtered_results.append({"href": href, "title": result.get("title", "")})

        # Fallback to top non-excluded links
        if not filtered_results:
            filtered_results = [
                {"href": result.get("href", ""), "title": result.get("title", "")}
                for result in results
                if (result.get("href") and
                    not any(result["href"].lower().endswith(ext) for ext in exclude_extensions) and
                    not any(x in result["href"].lower() for x in exclude_domains))
            ]

        logging.info(f"Scraped {len(filtered_results)} URLs for query: {query}")
        return filtered_results[:max_results]
    except Exception as e:
        logging.error(f"Error searching for '{query}': {e}")
        return []

def find_main_website(query: str) -> Union[str, None]:
    """Find the main website URL with strict verification using Google search, prioritizing root domain."""
    query_clean = clean_search_query(query)
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # Define preferred domains based on certificate type
        preferred_domains = [".edu.in", ".ac.in", ".gov.in", ".edu", ".org"] if any(x in query.lower() for x in ["university", "board", "institute"]) else [".com", ".co", ".org"]
        org_tokens = query_clean.split()
        results = google_search_with_retry(query_clean, num_results=10)
        exclude_extensions = [".pdf", ".doc", ".docx"]
        exclude_domains = ["linkedin.com", "facebook.com", "twitter.com", "instagram.com", "glassdoor.com", "wikipedia.org", "youtube.com"]
        exclude_subpaths = ["/about", "/blog", "/news", "/article", "/post", "/contact"]

        # Helper function to check if URL is a homepage
        def is_homepage(url):
            parsed = urllib.parse.urlparse(url)
            path = parsed.path.lower().rstrip('/')
            return path == '' or path == '/' or path in ['/index', '/home', '/homepage']

        # Strict verification with title and content check
        for result in results:
            href = result.get("href", "")
            if (href and not any(href.lower().endswith(ext) for ext in exclude_extensions) and
                not any(x in href.lower() for x in exclude_domains)):
                # Prioritize homepage URLs
                if is_homepage(href) and any(domain in href.lower() for domain in preferred_domains):
                    try:
                        response = session.get(href, headers=get_random_headers(), timeout=15)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.text, 'html.parser')
                        page_text = soup.get_text().upper()
                        title = soup.title.string.upper() if soup.title else ""
                        # Require 80% token match for higher accuracy
                        matched_tokens = sum(1 for token in org_tokens if token in page_text or token in title)
                        if matched_tokens >= len(org_tokens) * 0.8:
                            logging.info(f"Found main website for {query}: {href} (preferred domain, homepage)")
                            return href.split("?")[0].rstrip('/')
                    except Exception as e:
                        logging.warning(f"Failed to verify {href}: {e}")
                        continue

        # Fallback with stricter matching, still prioritizing homepage
        for result in results:
            href = result.get("href", "")
            if (href and not any(href.lower().endswith(ext) for ext in exclude_extensions) and
                not any(x in href.lower() for x in exclude_domains) and
                not any(subpath in href.lower() for subpath in exclude_subpaths)):
                try:
                    response = session.get(href, headers=get_random_headers(), timeout=15)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    page_text = soup.get_text().upper()
                    title = soup.title.string.upper() if soup.title else ""
                    matched_tokens = sum(1 for token in org_tokens if token in page_text or token in title)
                    if matched_tokens >= len(org_tokens) * 0.8:
                        logging.info(f"Found main website for {query}: {href} (fallback)")
                        return href.split("?")[0].rstrip('/')
                except Exception as e:
                    logging.warning(f"Failed to verify {href}: {e}")
                    continue

        logging.info(f"No valid main website found for query: {query}")
        return None
    except Exception as e:
        logging.error(f"Error finding main website for '{query}': {e}")
        return None

# === Web Content Extraction ===
def extract_intro_from_url(url):
    """Fetch raw text from page with retries."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        response = session.get(url, headers=get_random_headers(), timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ""
        for p in paragraphs:
            line = p.get_text().strip()
            if len(line) > 30:
                text += line + " "
            if len(text) > 1500:
                break
        logging.info(f"Extracted text from {url}")
        return text.strip()
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        print(f"⚠️ Error fetching {url}: {e}")
        return ""

def summarize_about_info(raw_text, llm, certificate_type):
    """Summarize scraped text into a 2-3 sentence About Us description."""
    org_type_hint = "educational board, university, or institution" if certificate_type in ["10th", "12th", "bachelor", "master"] else "company or organization"
    prompt = f"""
You are given raw content scraped from an {org_type_hint}'s website.
Extract 2-3 clear sentences that best describe:
- What the {org_type_hint} does (e.g., conducts exams, provides education).
- Its mission, field, services, or core focus.
Focus only on the specific organization, not general education systems or courses.
If factual details are missing, provide a concise, factual introduction based on the content.
Content:
\"\"\"{raw_text}\"\"\"
Respond ONLY with the extracted summary.
"""
    resp = llm.complete(prompt)
    logging.info("Summarized About Us")
    return resp.text.strip()

def get_org_intro_summary(organization_name: str, llm, certificate_type: str) -> Union[Dict, None]:
    """Search, scrape, and summarize About Us info."""
    print(f"🔍 Scraping About Us for: {organization_name}")
    logging.info(f"Scraping About Us for: {organization_name}")
    query = f"{clean_search_query(organization_name)} About Us"
    logging.info(f"Search query: {query}")
    search_results = search_top_urls(query)
    urls = [r["href"] for r in search_results if "href" in r]

    print(f"📜 Top {min(3, len(urls))} URLs to scrape for {organization_name}:")
    if urls:
        for i, url in enumerate(urls[:3], 1):
            print(f"  {i}. {url}")
    else:
        print("  (No URLs found)")
    logging.info(f"Top {min(3, len(urls))} URLs for {organization_name}: {urls[:3]}")

    main_website = find_main_website(organization_name)
    all_texts = []

    for url in urls:
        raw_text = extract_intro_from_url(url)
        if raw_text:
            all_texts.append((url, raw_text))

    for url, raw_text in all_texts:
        summary = summarize_about_info(raw_text, llm, certificate_type)
        if summary:
            about_info = {
                "summary": summary,
                "source_urls": urls,
                "main_website": main_website,
                "organization": organization_name,
                "certificate_type": certificate_type
            }
            aboutus_cache[organization_name.upper().strip()] = about_info
            print(f"✅ Scraped About Us for: {organization_name}")
            print(f"\n🏛️ Organization/University: {organization_name}")
            print(f"🌐 Main Website: {about_info['main_website']}")
            print(f"📝 About Us Summary:\n{about_info['summary']}")
            print(f"🔗 Source URLs:\n{about_info['source_urls']}")
            logging.info(f"Scraped About Us for {organization_name}: {summary}")
            return about_info

    print(f"⚠️ Failed to find a good About Us for {organization_name}")
    logging.warning(f"Failed to find a good About Us for {organization_name}")
    return None

# === URL and Organization Extraction ===
def extract_url(text: str, org_name: str = None) -> Union[str, None]:
    """Extract a relevant URL from certificate text."""
    urls = re.findall(r'(https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s\)\]\},\.]+)', text)
    if not urls:
        return None

    if org_name:
        org_tokens = clean_search_query(org_name).split()
        for url in urls:
            url_lower = url.lower()
            try:
                result = urllib.parse.urlparse(url)
                if all([result.scheme, result.netloc]):
                    if any(token in url_lower for token in org_tokens) or any(domain in url_lower for domain in [".gov.in", ".edu.in", ".ac.in"]):
                        print(f"🔗 URL extracted from certificate: {url}")
                        logging.info(f"Extracted URL: {url}")
                        return url.strip()
            except ValueError:
                continue
    else:
        try:
            result = urllib.parse.urlparse(urls[0])
            if all([result.scheme, result.netloc]):
                print(f"🔗 URL extracted from certificate: {urls[0]}")
                logging.info(f"Extracted URL: {urls[0]}")
                return urls[0].strip()
        except ValueError:
            pass
    return None

def extract_organization_name(text: str, certificate_type: str, llm: LlamaCPP) -> Union[str, None]:
    """Extract the most accurate organization name using LLM, tailored to certificate type."""
    org_type = (
        "state or national education board (e.g., Karnataka Secondary Education Examination Board, CBSE)"
        if certificate_type in ["10th", "12th"]
        else "university (e.g., Visvesvaraya Technological University)"
        if certificate_type in ["bachelor", "master"]
        else "company (e.g., Technosoft Corporation)"
    )
    org_context = (
        "For 10th or 12th-grade certificates, select the education board responsible for conducting exams (e.g., Karnataka Secondary Education Examination Board for Karnataka), not schools, colleges, or universities like Visvesvaraya Technological University. "
        "For bachelor’s or master’s certificates, select the university awarding the degree (e.g., Visvesvaraya Technological University), not affiliated colleges or departments. "
        "For work or internship certificates, select the company or organization issuing the certificate. "
        "If multiple names appear, choose the one most likely to be the issuing organization based on the certificate type and context (e.g., in headers or prominent text)."
    )
    prompt = f"""
You are given OCR-extracted text from a {certificate_type} certificate, which may contain errors (e.g., 'TECHINOLOGICAL' instead of 'TECHNOLOGICAL', 'UNVIERSITY' instead of 'UNIVERSITY').
Extract the full name of the primary {org_type} that issued the certificate.
- {org_context}
- Correct any OCR errors by inferring the most likely correct name based on spelling patterns (e.g., 'UNVIERSITY' to 'UNIVERSITY').
- Return the clean, corrected name in UPPERCASE, excluding prefixes like 'CLEAN NAME:' or extraneous text.
Certificate Text:
\"\"\"{text}\"\"\"
Respond ONLY with the clean name, nothing else.
"""
    try:
        response = llm.complete(prompt)
        org_name = response.text.strip().upper()
        if org_name:
            print(f"🏷️ Organization name extracted by LLM: {org_name}")
            logging.info(f"Extracted organization name: {org_name}")
            return org_name
        return None
    except Exception as e:
        logging.error(f"LLM extraction failed: {e}")
        return None

async def handle_organization_aboutus(text: str, filename: str, llm: LlamaCPP, bachelor_info: list, master_info: list, work_info: dict) -> Union[dict, None]:
    """Process organization About Us info or main URL based on certificate type."""
    print(f"\n📄 Processing certificate: {filename}")
    logging.info(f"Processing certificate: {filename}")

    certificate_type = None
    filename_lower = filename.lower()

    if "10th" in filename_lower or "tenth" in filename_lower or "sslc" in filename_lower:
        certificate_type = "10th"
    elif "12th" in filename_lower or "twelfth" in filename_lower or "puc" in filename_lower:
        certificate_type = "12th"
    elif "bachelor" in filename_lower or "be" in filename_lower or "b.e" in filename_lower:
        certificate_type = "bachelor"
        if bachelor_info:
            print(f"⏩ Skipping bachelor certificate {filename}: bachelor_info already populated")
            logging.info(f"Skipping bachelor certificate {filename}: bachelor_info already populated")
            return None
    elif "master" in filename_lower or "me" in filename_lower or "m.e" in filename_lower:
        certificate_type = "master"
        if master_info:
            print(f"⏩ Skipping master certificate {filename}: master_info already populated")
            logging.info(f"Skipping master certificate {filename}: master_info already populated")
            return None
    elif "work" in filename_lower:
        certificate_type = "work"
    elif "internship" in filename_lower:
        certificate_type = "internship"

    if not certificate_type:
        print("❌ Unknown certificate type — skipping")
        logging.warning(f"Unknown certificate type: {filename}")
        return None

    # Extract organization name
    org_name = extract_organization_name(text, certificate_type, llm)
    if not org_name:
        print(f"⚠️ No organization name extracted for {certificate_type} certificate.")
        logging.warning(f"No organization name for {certificate_type}: {filename}")
        return None

    # Remove "CLEAN NAME:" prefix if present
    org_name = re.sub(r"^(clean name:\s*)", "", org_name, flags=re.IGNORECASE).strip()

    # Check for URL in certificate text
    url = extract_url(text, org_name)
    if url:
        print(f"✅ Institution/Company URL from certificate ({certificate_type}): {url}")
        logging.info(f"Institution/Company URL for {certificate_type}: {url}")
        return {"institution_url": url, "organization": org_name, "certificate_type": certificate_type}

    # Check cache
    cache_key = org_name.upper().strip()
    if cache_key in aboutus_cache:
        cache_source = aboutus_cache[cache_key].get("certificate_type", certificate_type)
        reuse_msg = f"♻️ Reusing cached info for {certificate_type}: {org_name}"
        if certificate_type in ["10th", "12th"] and cache_source != certificate_type:
            reuse_msg = f"♻️ Reusing cached info from {cache_source} for {certificate_type}: {org_name}"
        print(reuse_msg)
        logging.info(reuse_msg)
        about_info = aboutus_cache[cache_key]
        if certificate_type in ["10th", "12th"]:
            print(f"🌐 Main Website: {about_info['main_website']}")
            return {"institution_url": about_info['main_website'], "organization": org_name, "certificate_type": certificate_type}
        else:
            if certificate_type in ["work", "internship"]:
                work_info[org_name] = about_info
            print(f"\n🏛️ Organization/University: {org_name}")
            print(f"🌐 Main Website: {about_info['main_website']}")
            print(f"📝 About Us Summary:\n{about_info['summary']}")
            print(f"🔗 Source URLs:\n{about_info['source_urls']}")
            return about_info

    # For 10th and 12th, only fetch main URL
    if certificate_type in ["10th", "12th"]:
        main_website = find_main_website(org_name)
        if main_website:
            about_info = {
                "institution_url": main_website,
                "organization": org_name,
                "certificate_type": certificate_type
            }
            aboutus_cache[cache_key] = about_info
            print(f"🌐 Main Website: {main_website}")
            logging.info(f"Main website for {certificate_type}: {org_name} - {main_website}")
            return about_info
        else:
            print(f"⚠️ No main website found for {certificate_type}: {org_name}")
            logging.warning(f"No main website for {certificate_type}: {org_name}")
            return None

    # Scrape About Us for other certificate types
    about_info = get_org_intro_summary(organization_name=org_name, llm=llm, certificate_type=certificate_type)
    if about_info:
        aboutus_cache[cache_key] = about_info
        if certificate_type in ["work", "internship"]:
            work_info[org_name] = about_info
        print(f"Scraping newly...")
        print(f"\n🏛️ Organization/University: {org_name}")
        print(f"🌐 Main Website: {about_info['main_website']}")
        print(f"📝 About Us Summary:\n{about_info['summary']}")
        print(f"🔗 Source URLs:\n{about_info['source_urls']}")
        logging.info(f"About Us found for {certificate_type}: {org_name}")
        return about_info

    print(f"⚠️ No institution URL or About Us found for {certificate_type}: {org_name}")
    logging.warning(f"No institution URL or About Us for {certificate_type}: {org_name}")
    return None

# === OCR PDF ===
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PaddleOCR."""
    print(f"📄 Processing PDF: {pdf_path}")
    logging.info(f"Processing PDF: {pdf_path}")
    try:
        images = pdf2image.convert_from_path(pdf_path)
        texts = []
        for image in images:
            ocr_result = ocr.ocr(np.array(image), cls=True)
            if ocr_result and ocr_result[0]:
                page_text = " ".join(line[1][0] for line in ocr_result[0] if line[1])
                texts.append(page_text)
        text = "\n".join(texts).strip()
        logging.info(f"Extracted text from {pdf_path}")
        return text
    except Exception as e:
        print(f"⚠️ Error with {pdf_path}: {e}")
        logging.error(f"Error with {pdf_path}: {e}")
        return ""

# === Prompt Generator ===
def get_prompt(filename: str, _) -> str:
    """Generate LLM prompt based on filename."""
    filename_lower = filename.lower()
    if "10th" in filename_lower or "tenth" in filename_lower:
        return """Extract the following from the 10th Grade Certificate:
* Subjects:
* Total Percentage/GPA:
* Date of Passing:"""
    elif "12th" in filename_lower or "twelfth" in filename_lower:
        return """Extract the following from the 12th Grade Certificate:
* Subjects:
* Total Percentage/GPA:
* Date of Passing:"""
    elif "thesis" in filename_lower:
        return """Extract the following from the Technical or Thesis Certificate:
* Date of Publishing:
* Short summary about the thesis. Write it formally so as to mention in a resume:
* Institution or Company Name (if available):"""
    elif "b.e" in filename_lower or "bachelor" in filename_lower:
        if "8" in filename_lower or "viii" in filename_lower or "final" in filename_lower or "degree" in filename_lower:
            return """Extract the following from the Final B.E Certificate or 8th Semester:
* University Name:
* Major:
* CGPA/GPA:
* Date of Graduation (Year of Passing):
* 3 subject names from this semester:"""
        elif "5" in filename_lower or "v" in filename_lower:
            return """Extract 4 Subject Names without including the subject codes from B.E 5:
* Subjects:"""
        elif "6" in filename_lower or "vi" in filename_lower:
            return """Extract 4 Subject Names without including the subject codes from B.E 6:
* Subjects:"""
        elif "7" in filename_lower or "vii" in filename_lower:
            return """Extract 4 Subject Names without including the subject codes from B.E 7:
* Subjects:"""
        else:
            return "⏩ Skip processing — not a required semester or final certificate."
    elif "pg" in filename_lower or "master" in filename_lower:
        if "4" in filename_lower or "iv" in filename_lower or "final" in filename_lower or "degree" in filename_lower:
            return """Extract the following from the Final M.E Certificate or 4th Semester:
* University Name:
* Major:
* CGPA/GPA:
* Date of Graduation (Year of Passing):
* 3 subjects names from this semester:"""
        else:
            return """Extract 4 Subject Names without including the subject codes from M.E Semester:
* Subjects:"""
    elif "passport" in filename_lower:
        return """Extract the following from the Passport and format the address with appropriate punctuation:
* Full Name:
* Address:
* Date of Birth:"""
    elif "ielts" in filename_lower:
        return """Extract the following from the IELTS Certificate:
* Date of Passing:
* Total Score:
* Listening Score:
* Reading Score:
* Writing Score:
* Speaking Score:
* Remark as 'Proficient' if the candidate has passed. Remark:"""
    elif "toefl" in filename_lower:
        return """Extract the following from the TOEFL Certificate:
* Date of Passing:
* Total Score:
* Listening Score:
* Reading Score:
* Writing Score:
* Speaking Score:
* Remark as 'Proficient' if the candidate has passed. Remark:"""
    elif "german" in filename_lower:
        return """Extract the following from the German Language Certificate:
* Date of Passing:
* Total Score:
* Listening Score:
* Reading Score:
* Writing Score:
* Speaking Score:
* Remark as 'Proficient' if the candidate has passed. Remark:"""
    elif "work" in filename_lower:
        return """Extract the following from the Work Experience Letter:
* Company Name:
* Job Title:
* Start Date:
* End Date:"""
    elif "internship" in filename_lower:
        return """Extract the following from the Internship Certificate:
* Company Name:
* Job Title:
* Employer:
* Start Date:
* End Date:"""
    else:
        return "Extract all relevant information from this document."

# === Utility Functions ===
def find_documents_folder(root_dir: str) -> Union[str, None]:
    """Find a documents folder in the directory tree."""
    for root, dirs, _ in os.walk(root_dir):
        for d in dirs:
            if "document" in d.lower():
                return os.path.join(root_dir, d)
    return None

def find_student_cv_folder(root_dir: str) -> Union[str, None]:
    """Find a folder containing 'CV' in its name in the directory tree."""
    for root, dirs, _ in os.walk(root_dir):
        for d in dirs:
            if "cv" in d.lower():
                return os.path.join(root, d)
    return None

def find_tabulated_cv(cv_folder: str) -> Union[str, None]:
    """Recursively search for a PDF containing 'tabulated' in its name within the CV folder."""
    for root, _, files in os.walk(cv_folder):
        for f in files:
            if f.lower().endswith(".pdf") and "tabulated" in f.lower():
                return os.path.join(root, f)
    return None

def normalize_filename(filename: str) -> str:
    """Normalize filename for deduplication."""
    name = os.path.splitext(filename)[0].lower()
    name = re.sub(r'\s+', '', name)
    name = re.sub(r'\(\d+\)', '', name)
    name = re.sub(r'copy', '', name)
    return name.strip()

def process_tabulated_cv(pdf_path: str, llm: LlamaCPP, embed_model: HuggingFaceEmbedding, candidate: str) -> dict:
    """Extract specific information from the CV using Apache Tika with split queries."""
    print(f"📄 Processing CV: {pdf_path}")
    logging.info(f"Processing CV: {pdf_path}")

    try:
        parsed = parser.from_file(pdf_path, requestOptions={'timeout': 300})
        text = parsed.get('content', '').strip()
        if not text:
            print(f"No text extracted from: {pdf_path}")
            logging.warning(f"No text extracted from: {pdf_path}")
            return {}
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\n+', ' ', text).strip()
        print(f"Extracted text (first 200 chars): {text}...")
        logging.info(f"Extracted text: {text[:200]}...")
    except Exception as e:
        print(f"⚠️ Error with Tika extraction for {pdf_path}: {e}")
        logging.error(f"Error with Tika extraction for {pdf_path}: {e}")
        print(f"Falling back to PaddleOCR for {pdf_path}")
        logging.info(f"Falling back to PaddleOCR for {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"PaddleOCR failed for {pdf_path}")
            logging.error(f"PaddleOCR failed for {pdf_path}")
            return {}
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\n+', ' ', text).strip()

    result = {
        "work_experience": [],
        "professional_training": [],
        "computer_skills": [],
        "projects": [],
        "extracurricular_achievements": [],
        "hobbies": []
    }

    prompts = {
        "work_experience": """
Extract work/internship experience
- For each:
  - Company: Name of company/organization.
  - Role: Job title.
  - Time Period: Duration (e.g., "Jan 2020 - Dec 2021").
  - Responsibilities: Tasks performed.
Return JSON:
{{
  "work_experience": [
    {{
      "company": "",
      "role": "",
      "time_period": "",
      "responsibilities": []
    }}
  ]
}}
Text:
\"\"\"{text}\"\"\"
""",
        "professional_training": """
Extract professional training:
- List all training courses mentioned as strings.
Return JSON:
{{
  "professional_training": []
}}
Text:
\"\"\"{text}\"\"\"
""",
        "computer_skills": """
Extract computer/technical skills:
- List all skills.
Return JSON:
{{
  "computer_skills": []
}}
Text:
\"\"\"{text}\"\"\"
""",
        "projects": """
Extract project titles:
Return JSON:
{{
  "projects": []
}}
Text:
\"\"\"{text}\"\"\"
""",
        "extracurricular_hobbies": """
Extract:
- Extracurricular Achievements: Activities/achievements.
- Hobbies: Hobbies.
Return JSON:
{{
  "extracurricular_achievements": [],
  "hobbies": []
}}
Text:
\"\"\"{text}\"\"\"
"""
    }

    doc = Document(text=text, metadata={"filename": os.path.basename(pdf_path), "candidate": candidate})
    index = VectorStoreIndex.from_documents([doc], embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)

    max_attempts = 3
    for section, prompt in prompts.items():
        print(f"🔍 Querying section: {section}")
        logging.info(f"Querying section: {section}")
        response_text = ""
        for attempt in range(max_attempts):
            try:
                response = query_engine.query(prompt.format(text=text))
                response_text = response.response.strip()
                print(f"📄 Extracted {section} (attempt {attempt + 1}):\n{response_text}")
                logging.info(f"Extracted {section} (attempt {attempt + 1}): {response_text}")

                if not response_text.endswith('}'):
                    print(f"⚠️ Response truncated for {section} in attempt {attempt + 1}")
                    logging.warning(f"Response truncated for {section} in attempt {attempt + 1}")
                    if attempt == max_attempts - 1:
                        print(f"⚠️ Failed to extract {section} after {max_attempts} attempts")
                        # Fallback: Attempt to parse partial JSON
                        try:
                            partial_response = response_text + ']}' if '"work_experience": [' in response_text else response_text + ']}' if '"professional_training": [' in response_text else response_text + '}'
                            section_result = json.loads(partial_response)
                            print(f"✅ Recovered partial {section}: {section_result}")
                            logging.info(f"Recovered partial {section}: {section_result}")
                            for key in section_result:
                                result[key].extend(section_result[key])
                            break
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Failed to recover partial {section}: {e}")
                            logging.error(f"Failed to recover partial {section}: {e}")
                            continue
                    continue

                section_result = json.loads(response_text)
                print(f"✅ Parsed {section}: {section_result}")
                logging.info(f"Parsed {section}: {section_result}")

                for key in section_result:
                    result[key].extend(section_result[key])
                break
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing {section} response: {e}")
                logging.error(f"Error parsing {section} response: {e}")
                logging.error(f"Failed response text: {response_text}")
                if attempt == max_attempts - 1:
                    print(f"⚠️ Failed to extract {section} after {max_attempts} attempts")
                    # Fallback: Attempt to parse partial JSON
                    try:
                        partial_response = response_text + ']}' if '"work_experience": [' in response_text else response_text + ']}' if '"professional_training": [' in response_text else response_text + '}'
                        section_result = json.loads(partial_response)
                        print(f"✅ Recovered partial {section}: {section_result}")
                        logging.info(f"Recovered partial {section}: {section_result}")
                        for key in section_result:
                            result[key].extend(section_result[key])
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to recover partial {section}: {e}")
                        logging.error(f"Failed to recover partial {section}: {e}")

    print(f"✅ Final extracted CV info: {result}")
    logging.info(f"Final extracted CV info: {result}")
    return result

# === Main Workflow ===
async def main():
    """Main workflow to process candidate certificates."""
    shared_folder_path = "/content/drive/MyDrive/single"
    candidate_folders = sorted(os.listdir(shared_folder_path))
    seen_hashes = set()
    seen_cert_names = set()
    bachelor_info = []
    master_info = []
    work_info = {}
    cv_info = {}
    # Clear cache to avoid reusing incorrect URLs
    global aboutus_cache
    aboutus_cache = {}

    print(f"Bachelor Info: {bachelor_info}")
    print(f"Master Info: {master_info}")
    print(f"Work Info: {work_info}")

    for candidate in candidate_folders:
        candidate_path = os.path.join(shared_folder_path, candidate)
        if not os.path.isdir(candidate_path):
            continue

        print(f"\n👤 Processing candidate: {candidate}")

        # Process Student CV folder
        student_cv_path = find_student_cv_folder(candidate_path)
        if student_cv_path:
            print(f"📁 Found CV folder: {student_cv_path}")
            cv_path = find_tabulated_cv(student_cv_path)
            if cv_path:
                cv_data = process_tabulated_cv(cv_path, llm, embed_model, candidate)
                cv_info[candidate] = cv_data
                print(f"✅ Processed Tabulated CV for {candidate}: {cv_data}")
                logging.info(f"Processed Tabulated CV for {candidate}: {cv_data}")
            else:
                print(f"⚠️ No Tabulated CV PDF found in {student_cv_path}")
                logging.warning(f"No Tabulated CV PDF found in: {student_cv_path}")
        else:
            print(f"📁 No CV folder found in: {candidate}")
            logging.warning(f"No CV folder in: {candidate}")

        # Process documents folder
        documents_path = find_documents_folder(candidate_path)
        if not documents_path:
            print(f"📁 No documents folder found in: {candidate}")
            logging.warning(f"No documents folder in: {candidate}")
            continue

        pdf_files = sorted(f for f in os.listdir(documents_path) if f.lower().endswith(".pdf"))
        if not pdf_files:
            print("📄 No PDFs found.")
            logging.warning(f"No PDFs found for {candidate}")
            continue

        print(f"List of documents: {pdf_files}")

        for filename in pdf_files:
            pdf_path = os.path.join(documents_path, filename)

            # Determine certificate type early
            certificate_type = None
            filename_lower = filename.lower()
            if "10th" in filename_lower or "tenth" in filename_lower or "sslc" in filename_lower:
                certificate_type = "10th"
            elif "12th" in filename_lower or "twelfth" in filename_lower or "puc" in filename_lower:
                certificate_type = "12th"
            elif "bachelor" in filename_lower or "be" in filename_lower or "b.e" in filename_lower:
                certificate_type = "bachelor"
            elif "master" in filename_lower or "me" in filename_lower or "m.e" in filename_lower:
                certificate_type = "master"
            elif "work" in filename_lower:
                certificate_type = "work"
            elif "internship" in filename_lower:
                certificate_type = "internship"

            # Skip early B.E. semesters (1-4 or I-IV)
            if re.search(
    r"(b\.e|bachelor).*(1st|2nd|3rd|4th|first|second|third|fourth|1|2|3|4|i|ii|iii|iv)\s*(sem|s|semester)?\b",
    filename_lower,
    re.IGNORECASE
):
                print(f"⏩ Skipping early semester B.E certificate (Sem 1-4): {filename}")
                logging.info(f"Skipping early semester B.E (Sem 1-4): {filename}")
                continue

            # Skip B.E. project-related certificates
            if ("b.e" in filename_lower or "bachelor" in filename_lower) and re.search(
                r"(project|_project|-project)\b", filename_lower, re.IGNORECASE
            ):
                print(f"⏩ Skipping B.E project certificate: {filename}")
                logging.info(f"Skipping B.E project certificate: {filename}")
                continue

            # Skip M.E project-related certificates
            if ("m.e" in filename_lower or "master" in filename_lower) and re.search(
                r"(project|_project|-project)\b", filename_lower, re.IGNORECASE
            ):
                print(f"⏩ Skipping B.E project certificate: {filename}")
                logging.info(f"Skipping B.E project certificate: {filename}")
                continue



            # Skip sensitive or unneeded documents before extraction
            skip_keywords = ["letter of recommendation", "_lor", "-lor", " lor", "aadhaar", "aadhar", "pancard", "pan card", "technical", "moi", "extra", "training"]
            if any(keyword in filename_lower for keyword in skip_keywords):
                print(f"Skipping sensitive or unneeded document: {filename}")
                logging.info(f"Skipping document: {filename}")
                continue

            # Check for duplicates
            normalized_name = normalize_filename(filename)
            if normalized_name in seen_cert_names:
                print(f"⏩ Skipping duplicate certificate name: {filename}")
                logging.info(f"Skipping duplicate: {filename}")
                continue

            text = extract_text_from_pdf(pdf_path)
            if not text:
                print(f"⚠️ Skipping empty or failed file: {filename}")
                logging.error(f"Empty or failed: {filename}")
                continue

            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                print(f"⏩ Skipping duplicate content: {filename}")
                logging.info(f"Skipping duplicate content: {filename}")
                continue

            seen_cert_names.add(normalized_name)
            seen_hashes.add(content_hash)

            print(f"🔍 Extracted (first 200 chars): {text[:200]}...")
            logging.info(f"Extracted text (first 200 chars): {text[:200]}")

            doc = Document(text=text, metadata={"filename": filename, "candidate": candidate})
            index = VectorStoreIndex.from_documents([doc], embed_model=embed_model)
            query_engine = index.as_query_engine(llm=llm)

            prompt = get_prompt(filename, text)
            if "⏩ Skip processing" in prompt:
                print(f"⏩ Skipping non-required certificate: {filename}")
                logging.info(f"Skipping non-required certificate: {filename}")
                continue

            response = query_engine.query(prompt)
            response_text = response.response.strip()

            # Cross-reference with CV info
            if candidate in cv_info and cv_info[candidate]:
                if "ieee" in filename_lower and "Published Paper in IEEE" in cv_info[candidate].get("extracurricular_achievements", []):
                    print(f"✅ Certificate {filename} matches CV achievement: Published Paper in IEEE")
                    logging.info(f"Certificate {filename} matches CV achievement: Published Paper in IEEE")
                if certificate_type in ["work", "internship"]:
                    org_name = extract_organization_name(text, certificate_type, llm)
                    if org_name and org_name.lower() in [resp.lower() for resp in cv_info[candidate].get("work_responsibilities", [])]:
                        print(f"✅ Certificate {filename} matches CV work/internship: {org_name}")
                        logging.info(f"Certificate {filename} matches CV work/internship: {org_name}")

            # Process About Us for relevant certificate types
            if certificate_type in ["10th", "12th", "bachelor", "master", "work", "internship"]:
                await handle_organization_aboutus(text, filename, llm, bachelor_info, master_info, work_info)
            else:
                print(f"⏩ Skipping About Us processing for: {filename}")
                logging.info(f"Skipping About Us processing for: {filename}")

            print(f"\n✅ {candidate} | {filename} → {response_text}\n{'-'*60}")
            logging.info(f"Processed {candidate} | {filename}: {response_text}")

        # Print CV info for the candidate
        if candidate in cv_info and cv_info[candidate]:
            print(f"\n📝 CV Information for {candidate}:")
            print(f"Work Responsibilities: {cv_info[candidate].get('work_responsibilities', [])}")
            print(f"Professional Training: {cv_info[candidate].get('professional_training', [])}")
            print(f"Computer Skills: {cv_info[candidate].get('computer_skills', [])}")
            print(f"Projects: {cv_info[candidate].get('projects', [])}")
            print(f"Extracurricular Achievements: {cv_info[candidate].get('extracurricular_achievements', [])}")
            print(f"Hobbies: {cv_info[candidate].get('hobbies', [])}")
            logging.info(f"CV info for {candidate}: {cv_info[candidate]}")

if __name__ == "__main__":
    asyncio.run(main())
