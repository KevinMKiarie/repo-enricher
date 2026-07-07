import os
import json
import time
import base64
import requests
from pathlib import Path

PROGRESS_FILE = '.github/repo_progress.json'
MODELS_URL = 'https://models.inference.ai.azure.com/chat/completions'
GH_API = 'https://api.github.com'
MODEL = 'gpt-4o-mini'

gh_headers = {
    'Authorization': f'Bearer {os.environ["GH_PAT"]}',
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}
ai_headers = {
    'Authorization': f'Bearer {os.environ["GITHUB_TOKEN"]}',
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
    return base64.b64decode(r.json()['content']).decode('utf-8', errors='ignore')[:2000]


def ai_generate(repo_name, readme):
    context = f'Repository name: {repo_name}'
    if readme:
        context += f'\n\nREADME excerpt:\n{readme}'

    prompt = (
        f'{context}\n\n'
        'Generate a concise GitHub repository description (max 120 characters) '
        'and up to 5 relevant topic tags (lowercase, hyphenated, no spaces).\n'
        'Reply ONLY with valid JSON in this exact format:\n'
        '{"description": "...", "topics": ["tag1", "tag2"]}'
    )

    r = requests.post(MODELS_URL, headers=ai_headers, json={
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.3,
        'max_tokens': 150,
    })

    if r.status_code == 429:
        return None, 'rate_limited'

    r.raise_for_status()
    text = r.json()['choices'][0]['message']['content']
    data = json.loads(text[text.index('{'):text.rindex('}') + 1])
    return data, 'ok'


def update_repo(owner, repo, description, topics):
    requests.patch(
        f'{GH_API}/repos/{owner}/{repo}',
        headers=gh_headers,
        json={'description': description},
    )
    requests.put(
        f'{GH_API}/repos/{owner}/{repo}/topics',
        headers=gh_headers,
        json={'names': topics},
    )


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
        print(f'Processing {full_name} ...', end=' ', flush=True)

        readme = get_readme(owner, name)
        result, status = ai_generate(name, readme)

        if status == 'rate_limited':
            print('\nRate limit reached — progress saved. Will resume on next run.')
            save_progress(progress)
            raise SystemExit(0)

        update_repo(owner, name, result['description'], result['topics'])
        done.add(full_name)
        progress['done'] = sorted(done)
        save_progress(progress)
        print(f'done  [{result["description"]}] tags={result["topics"]}')

        time.sleep(1)

    print('All repositories processed.')


main()
