import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def load_csv_data(file_path: str, text_columns: List[str], metadata_columns: List[str]) -> List[Document]:
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except:
            df = pd.read_csv(file_path, encoding='cp1252')
    
    documents = []
    
    for idx, row in df.iterrows():
        text_parts = []
        
        # Laureate/Organization name
        if pd.notna(row.get('fullName')):
            text_parts.append(f"Laureate: {row['fullName']}")
        elif pd.notna(row.get('orgName')):
            text_parts.append(f"Organization: {row['orgName']}")

        # Prize information
        if pd.notna(row.get('awardYear')):
            text_parts.append(f"Year: {row['awardYear']}")
        if pd.notna(row.get('category')):
            text_parts.append(f"Category: {row['category']}")
        if pd.notna(row.get('categoryFullName')):
            text_parts.append(f"Category Full Name: {row['categoryFullName']}")
        if pd.notna(row.get('motivation')):
            text_parts.append(f"Motivation: {row['motivation']}")
        if pd.notna(row.get('prizeAmount')):
            text_parts.append(f"Prize Amount: {row['prizeAmount']} SEK")
        if pd.notna(row.get('dateAwarded')):
            text_parts.append(f"Date Awarded: {row['dateAwarded']}")
        
        # Personal information
        if pd.notna(row.get('gender')):
            text_parts.append(f"Gender: {row['gender']}")
        if pd.notna(row.get('birth_date')):
            text_parts.append(f"Birth Date: {row['birth_date']}")
        if pd.notna(row.get('birth_city')):
            birth_info = f"Birth Place: {row['birth_city']}"
            if pd.notna(row.get('birth_country')):
                birth_info += f", {row['birth_country']}"
            text_parts.append(birth_info)
        
        # Organizational information
        if pd.notna(row.get('orgName')):
            if pd.notna(row.get('acronym')):
                text_parts.append(f"Acronym: {row['acronym']}")
            if pd.notna(row.get('org_founded_date')):
                text_parts.append(f"Founded: {row['org_founded_date']}")
        
        # Affiliations
        for i in range(1, 5):
            col = f'affiliation_{i}'
            if col in row and pd.notna(row[col]):
                text_parts.append(f"Affiliation {i}: {row[col]}")
        
        # Type of laureate
        if pd.notna(row.get('ind_or_org')):
            text_parts.append(f"Type: {row['ind_or_org']}")
        
        if text_parts:
            combined_text = "\n".join(text_parts)
            
            metadata = {}
            for col in metadata_columns:
                if col in row and pd.notna(row[col]):
                    metadata[col] = str(row[col])
            
            metadata["row_index"] = idx
            metadata["source"] = str(file_path)
            
            doc = Document(page_content=combined_text, metadata=metadata)
            documents.append(doc)
    
    print(f"Created {len(documents)} documents from CSV")
    return documents

def create_text_splitter(chunk_size: int, chunk_overlap: int):

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )