for old_ref in sorted(old_refs):
    old_url = old[old_ref].get("URL")
    # Exact match by reference
    if old_ref in new:
        rec = new[old_ref].copy()
        # --- Preserve manual edits for key fields ---
        fields_to_preserve = ["Unit Type", "Unit Number", "Type"]
        for field in fields_to_preserve:
            if old[old_ref].get(field):
                rec[field] = old[old_ref][field]
        rec["Availability"] = "Available"
        rec["_color"] = None
        merged.append((old_ref, rec))
        processed_new_refs.add(old_ref)
    # Fallback: URL match with a different new reference
    elif old_url in url_to_ref:
        new_ref = url_to_ref[old_url]
        rec = new[new_ref].copy()
        # --- Preserve manual edits for key fields ---
        fields_to_preserve = ["Unit Type", "Unit Number", "Type"]
        for field in fields_to_preserve:
            if old[old_ref].get(field):
                rec[field] = old[old_ref][field]
        rec["Availability"] = "Available"
        rec["_color"] = None
        # Preserve the original reference code
        rec["Reference Number"] = old_ref
        merged.append((old_ref, rec))
        processed_new_refs.add(new_ref)
    else:
        rec = old[old_ref].copy()
        rec["Availability"] = "Not available"
        rec["_color"] = "removed"
        merged.append((old_ref, rec)) 