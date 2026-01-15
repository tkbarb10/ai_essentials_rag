import re

def structure_markdown(text):
    # 1. Convert bold Chapters to # (Level 1)
    text = re.sub(r"^\s*\*\*CHAPTER\s+(\d+)\*\*\s*$", r"# CHAPTER \1", text, flags=re.MULTILINE | re.IGNORECASE)

    # 2. Fix Broken/Multi-line Blueprints and Convert to ### (Level 3)
    # This regex looks for a bold Blueprint line, and then checks if the IMMEDIATELY 
    # following line is also a bold line. If so, it merges them.
    # Pattern: **Blueprint: ...** \n ** ... **
    def merge_blueprints(match):
        part1 = match.group(1).strip()
        part2 = match.group(2).strip()
        # Merge them into one clean line
        return f"### {part1} {part2}"

    # We run this in a loop or with a specific pattern to catch the 2-line bold split
    text = re.sub(r"^\s*\*\*([Bb]lueprint:.*?)\*\*\s*\n\s*\*\*(.*?)\*\*\s*$", 
                  merge_blueprints, 
                  text, 
                  flags=re.MULTILINE)

    # 3. Catch any remaining Single-line Blueprints
    text = re.sub(r"^\s*\*\*([Bb]lueprint:.*?)\*\*\s*$", r"### \1", text, flags=re.MULTILINE)

    # 4. Convert Standalone Bold Capitalized Titles to ## (Level 2)
    # We use a negative lookahead to ensure we don't catch the newly created ### headers
    text = re.sub(r"^\s*\*\*((?![Bb]lueprint)[A-Z].*?)\*\*\s*$", r"## \1", text, flags=re.MULTILINE)

    # 5. Downgrade 4-hash headers to ## (Level 2)
    text = re.sub(r"^#### (.*?)$", r"## \1", text, flags=re.MULTILINE)

    return text