import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path


DEFAULT_ROOTS = [
    Path(r"C:\Users\YU\.codex\skills"),
    Path(r"C:\Users\YU\.agents\skills"),
    Path(r"C:\Users\YU\.codex\plugins\cache"),
]


def parse_front_matter(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    name = ""
    description = ""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            header = text[3:end]
            for line in header.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip().strip('"').strip("'")
                if key.strip() == "name":
                    name = value
                elif key.strip() == "description":
                    description = value
    if not name:
        name = path.parent.name
    return name, re.sub(r"\s+", " ", description).strip()


def discover_skills(roots):
    rows = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for skill_file in sorted(root.rglob("SKILL.md")):
            resolved = str(skill_file)
            if resolved in seen:
                continue
            seen.add(resolved)
            name, description = parse_front_matter(skill_file)
            if ".codex\\plugins\\cache" in resolved:
                source = "plugin-cache"
            elif ".codex\\skills\\.system" in resolved:
                source = "system-skill"
            elif ".codex\\skills" in resolved:
                source = "personal-skill"
            elif ".agents\\skills" in resolved:
                source = "agent-skill"
            else:
                source = "unknown"
            rows.append(
                {
                    "name": name,
                    "description": description,
                    "source": source,
                    "path": resolved,
                }
            )
    rows.sort(key=lambda item: (item["source"], item["name"].lower(), item["path"].lower()))
    return rows


def write_markdown(rows, output):
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Local Codex Skills Inventory",
        "",
        f"Generated: {generated}",
        f"Total skills: {len(rows)}",
        "",
        "| # | Skill | Source | Description | Path |",
        "|---:|---|---|---|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        desc = row["description"].replace("|", "\\|")
        path = row["path"].replace("|", "\\|")
        lines.append(f"| {idx} | `{row['name']}` | {row['source']} | {desc} | `{path}` |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Inventory local Codex SKILL.md files.")
    parser.add_argument(
        "--output",
        default=r"D:\luolin\V13\local_reports\local_skills_inventory.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--json-output",
        default=r"D:\luolin\V13\local_reports\local_skills_inventory.json",
        help="JSON output path.",
    )
    args = parser.parse_args()

    rows = discover_skills(DEFAULT_ROOTS)
    output = Path(args.output)
    json_output = Path(args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(rows, output)
    json_output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"skills={len(rows)}")
    print(f"markdown={output}")
    print(f"json={json_output}")


if __name__ == "__main__":
    main()
