def merge_docs(docs):
    if not docs:
        return []

    merged_docs = []
    current_heading = None
    current_doc = None

    for doc in docs:
        # Check if doc is None
        if doc is None:
            print("Warning: Found None document, skipping...")
            continue
            
        # Check if doc has required attributes
        if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
            print(f"Warning: Document missing required attributes, skipping...")
            continue
            
        # Get heading from the correct nested location
        dl_meta = doc.metadata.get("dl_meta")
        if dl_meta and dl_meta.get('headings'):
            heading = dl_meta.get('headings')[0]
        else:
            heading = None
        
        # If this is a new heading, start a new merged document
        if heading != current_heading:
            if current_doc is not None:
                merged_docs.append(current_doc)
            
            # Create a copy of the document instead of using reference
            current_doc = type(doc)(
                page_content=doc.page_content,
                metadata=doc.metadata.copy()
            )
            current_heading = heading
        else:
            # For same heading, merge content while handling the heading prefix
            content = doc.page_content
            if content and "\n" in content:
                # Remove heading prefix if present
                content = content.split("\n", 1)[1]
            
            # Only append if content exists
            if content:
                current_doc.page_content += "\n" + content

    # Add the last document
    if current_doc is not None:
        merged_docs.append(current_doc)

    return merged_docs