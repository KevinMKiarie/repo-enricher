import os
import json
import time
import base64
import requests
from pathlib import Path

PROGRESS_FILE = '.github/repo_progress.json'
MODELS_URL = 'https://api.groq.com/openai/v1/chat/completions'
GH_API = 'https://api.github.com'
MODEL = 'llama-3.3-70b-versatile'

# Manifest files that best describe a project's purpose and dependencies
MANIFEST_FILES = [
    'package.json', 'pyproject.toml', 'setup.cfg', 'setup.py',
    'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle', 'composer.json',
    'pubspec.yaml', 'mix.exs', 'Gemfile', 'build.sbt', 'Makefile', 'Dockerfile',
]

SOURCE_EXTENSIONS = {
    '.py', '.js', '.ts', '.go', '.rs', '.java', '.rb', '.php',
    '.swift', '.kt', '.c', '.cpp', '.cs', '.sh',
}

gh_headers = {
    'Authorization': f'Bearer {os.environ["GH_PAT"]}',
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}
ai_headers = {
    'Authorization': f'Bearer {os.environ["GROQ_API_KEY"]}',
    'Content-Type': 'application/json',
}


def load_progress():
    p = Path(PROGRESS_FILE)
    return json.loads(p.read_text()) if p.exists() else {'done': []}


def save_progress(progress):
    Path(PROGRESS_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(PROGRESS_FILE).write_text(json.dumps(progress, indent=2))


def list_repos():
    repos, page = [], 1
    while True:
        r = requests.get(
            f'{GH_API}/user/repos',
            headers=gh_headers,
            params={'per_page': 100, 'page': page, 'type': 'owner'},
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        repos.extend(batch)
        page += 1
    return repos


def get_readme(owner, repo):
    r = requests.get(f'{GH_API}/repos/{owner}/{repo}/readme', headers=gh_headers)
    if r.status_code != 200:
        return ''
    return base64.b64decode(r.json()['content']).decode('utf-8', errors='ignore')[:1500]


def get_file_tree(owner, repo, branch):
    """Fetch the full file tree for a repo branch (blobs only, skip large files)."""
    r = requests.get(
        f'{GH_API}/repos/{owner}/{repo}/git/trees/{branch}',
        headers=gh_headers,
        params={'recursive': '1'},
    )
    if r.status_code != 200:
        return []
    items = r.json().get('tree', [])
    return [
        item['path'] for item in items
        if item['type'] == 'blob' and item.get('size', 0) < 100_000
    ]


def get_file_content(owner, repo, path, max_chars=700):
    """Fetch and decode a single file's content via the GitHub Contents API."""
    r = requests.get(
        f'{GH_API}/repos/{owner}/{repo}/contents/{path}',
        headers=gh_headers,
    )
    if r.status_code != 200:
        return ''
    data = r.json()
    if data.get('encoding') == 'base64':
        return base64.b64decode(data['content']).decode('utf-8', errors='ignore')[:max_chars]
    return ''


def gather_project_context(owner, repo, default_branch, readme):
    """
    Build a rich context string by scanning the repo's file tree and fetching
    the most informative files: README, manifest/config files, and top-level
    source files. This gives the AI real understanding of the project's purpose.
    """
    files = get_file_tree(owner, repo, default_branch)
    file_set = set(files)
    parts = []

    if readme:
        parts.append(f'README:\n{readme}')

    # Fetch up to 2 manifest/config files (package.json, pyproject.toml, etc.)
    fetched_manifests = 0
    for mf in MANIFEST_FILES:
        if fetched_manifests >= 2:
            break
        if mf in file_set:
            content = get_file_content(owner, repo, mf, max_chars=700)
            if content:
                parts.append(f'{mf}:\n{content}')
                fetched_manifests += 1

    # Fetch up to 2 top-level source files for additional intent signals
    top_src = [
        f for f in files
        if '/' not in f and Path(f).suffix in SOURCE_EXTENSIONS
    ][:2]
    for sf in top_src:
        content = get_file_content(owner, repo, sf, max_chars=500)
        if content:
            parts.append(f'{sf} (excerpt):\n{content}')

    # Always include the top-level directory structure as a quick orientation
    if files:
        top_items = sorted({f.split('/')[0] for f in files})[:20]
        parts.append('Top-level structure:\n' + '\n'.join(top_items))

    return '\n\n---\n\n'.join(parts)


def ai_generate(repo_name, context):
    system_msg = (
        'You write GitHub repository descriptions. '
        'Your goal is to describe WHAT the project does and WHY it exists — '
        'its purpose, the problem it solves, or the value it provides. '
        'Do NOT just list the tech stack or programming language. '
        'Be specific and concrete. Keep descriptions under 120 characters.'
    )
    user_msg = (
        f'Repository: {repo_name}\n\n'
        f'{context}\n\n'
        'Based on the context above, generate a GitHub repository description and up to 5 topic tags.\n'
        'Reply ONLY with valid JSON in this exact format:\n'
        '{"description": "...", "topics": ["tag1", "tag2"]}'
    )

    r = requests.post(MODELS_URL, headers=ai_headers, json={
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_msg},
        ],
        'temperature': 0.3,
        'max_tokens': 200,
    })

    if r.status_code == 429:
        return None, 'rate_limited'

    if not r.ok:
        print(f'\nGroq error {r.status_code}: {r.text}')
        r.raise_for_status()

    text = r.json()['choices'][0]['message']['content']
    data = json.loads(text[text.index('{'):text.rindex('}') + 1])
    return data, 'ok'


def update_repo(owner, repo, description, topics):
    r1 = requests.patch(
        f'{GH_API}/repos/{owner}/{repo}',
        headers=gh_headers,
        json={'description': description},
    )
    r2 = requests.put(
        f'{GH_API}/repos/{owner}/{repo}/topics',
        headers=gh_headers,
        json={'names': topics},
    )
    if not r1.ok:
        print(f'\n  [WARN] description update failed {r1.status_code}: {r1.text}')
    if not r2.ok:
        print(f'\n  [WARN] topics update failed {r2.status_code}: {r2.text}')
    return r1.ok and r2.ok


def main():
    progress = load_progress()
    done = set(progress['done'])

    repos = list_repos()
    if not repos:
        print('No repositories found.')
        return

    owner = repos[0]['owner']['login']

    pending = [
        r for r in repos
        if r['full_name'] not in done
        and (not (r.get('description') or '').strip() or not r.get('topics', []))
    ]

    print(f'{len(pending)} repos to process, {len(done)} already done.')

    for repo in pending:
        name = repo['name']
        full_name = repo['full_name']
        default_branch = repo.get('default_branch', 'main')
        print(f'Processing {full_name} ...', end=' ', flush=True)

        readme = get_readme(owner, name)
        context = gather_project_context(owner, name, default_branch, readme)
        result, status = ai_generate(name, context)

        if status == 'rate_limited':
            print('\nRate limit reached — progress saved. Will resume on next run.')
            save_progress(progress)
            raise SystemExit(0)

        updated = update_repo(owner, name, result['description'], result['topics'])
        if updated:
            done.add(full_name)
            progress['done'] = sorted(done)
            save_progress(progress)
            print(f'done  [{result["description"]}] tags={result["topics"]}')
        else:
            print(f'skipped (update failed — see warning above)')

        time.sleep(1)

    print('All repositories processed.')


main()
