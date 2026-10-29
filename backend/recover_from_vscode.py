"""Restore missing PaperLens Python files from VS Code Local History.

Existing files, VS Code history, SQLite and PDFs are never overwritten.
Run with the API and worker stopped. Only paths inside --backend/app qualify.
"""

import argparse
import ast
from datetime import datetime
import json
import os
from pathlib import Path, PureWindowsPath
import zipfile
from urllib.parse import unquote, urlsplit


def _relative_source(resource, resource_root):
    if not isinstance(resource, str):
        return None
    parsed = urlsplit(resource)
    if parsed.scheme != 'file' or parsed.netloc not in ('', 'localhost'):
        return None
    path = PureWindowsPath(unquote(parsed.path).lstrip('/'))
    if not path.is_absolute() or '..' in path.parts:
        return None
    try:
        relative = path.relative_to(resource_root)
    except ValueError:
        return None
    if relative.suffix.lower() != '.py' or not relative.parts:
        return None
    return relative


def _safe_snapshot(folder, entry_id):
    if not isinstance(entry_id, str) or entry_id in ('', '.', '..'):
        return None
    if '/' in entry_id or '\\' in entry_id:
        return None
    path = folder / entry_id
    if not path.is_file() or path.is_symlink():
        return None
    return path


def restore(backend, history, resource_root, cutoff=None):
    backend = backend.resolve()
    app = backend / 'app'
    if not backend.is_dir() or not app.is_dir():
        raise RuntimeError('Backend must already contain an app directory. No main.py is required for recovery.')
    if not history.is_dir():
        raise RuntimeError(f'History directory not found: {history}')

    candidates = {}
    history_records = []
    warnings = []
    for metadata in sorted(history.rglob('entries.json')):
        try:
            info = json.loads(metadata.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        if not isinstance(info, dict):
            continue
        relative = _relative_source(info.get('resource'), resource_root)
        if relative is None:
            continue
        history_records.append(metadata)
        for entry in info.get('entries', []):
            if not isinstance(entry, dict):
                continue
            timestamp = entry.get('timestamp')
            if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
                continue
            if cutoff is not None and timestamp > cutoff:
                continue
            snapshot = _safe_snapshot(metadata.parent, entry.get('id'))
            if snapshot is None:
                continue
            key = str(relative).casefold()
            candidates.setdefault(key, []).append((timestamp, relative, snapshot))

    if not candidates:
        raise RuntimeError('No matching Python snapshots found. No source files were changed. Check --history and the original project path.')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = backend / f'vscode_recovery_backup_{stamp}.zip'
    with zipfile.ZipFile(backup, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(app.rglob('*')):
            if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
                archive.write(path, Path('app_current') / path.relative_to(app))
        written = set()
        for metadata in history_records:
            name = str(Path('vscode_history') / metadata.relative_to(history))
            if name not in written:
                archive.write(metadata, name)
                written.add(name)
        for versions in candidates.values():
            for _, _, snapshot in versions:
                name = str(Path('vscode_history') / snapshot.relative_to(history))
                if name not in written:
                    archive.write(snapshot, name)
                    written.add(name)

    restored, preserved, unavailable = [], [], []
    for key, versions in sorted(candidates.items()):
        versions.sort(key=lambda entry: entry[0], reverse=True)
        relative = versions[0][1]
        destination = app.joinpath(*relative.parts)
        try:
            destination.resolve().relative_to(app.resolve())
        except ValueError:
            warnings.append(f'Refused path outside app: {relative}')
            continue
        display = str(Path('app').joinpath(*relative.parts))
        if destination.exists():
            preserved.append(display)
            continue
        for timestamp, _, snapshot in versions:
            try:
                data = snapshot.read_bytes()
                # Parse source without executing it or importing the app.
                tree = ast.parse(data, filename=str(relative))
                if not tree.body and relative.name != '__init__.py':
                    raise ValueError('Empty source snapshot')
            except (OSError, SyntaxError, ValueError) as exc:
                warnings.append(f'Skipped invalid snapshot for {relative}: {type(exc).__name__}')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with destination.open('xb') as target:
                    target.write(data)
            except FileExistsError:
                preserved.append(display)
                break
            restored.append({
                'file': display,
                'history_snapshot': str(snapshot),
                'saved_at_local': datetime.fromtimestamp(timestamp / 1000).isoformat(),
            })
            print('RESTORED:', display)
            break
        else:
            unavailable.append(display)

    # Report which imported app modules are still absent. This is static only.
    missing = {}
    syntax_errors = []
    for path in sorted(app.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (OSError, SyntaxError, ValueError) as exc:
            syntax_errors.append({'file': str(path.relative_to(backend)), 'error': str(exc)})
            continue
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names if alias.name.startswith('app.')]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module and node.module.startswith('app.'):
                if node.module in ('app.schemas', 'app.services', 'app.repositories', 'app.adapters', 'app.controllers', 'app.core', 'app.models'):
                    modules = [node.module + '.' + alias.name for alias in node.names
                               if alias.name != '*' and alias.name[:1].islower()]
                else:
                    modules = [node.module]
            for module in modules:
                location = backend.joinpath(*module.split('.'))
                if not location.is_dir() and not location.with_suffix('.py').is_file():
                    missing.setdefault(module, set()).add(str(path.relative_to(backend)))
    critical = [name for name in ('main.py', 'review_worker.py', 'pdf_processing.py')
                if not (app / name).is_file()]
    report = {
        'backend': str(backend), 'history': str(history),
        'backup': str(backup),
        'restored_missing_files': restored,
        'existing_files_preserved': preserved,
        'no_valid_snapshot': unavailable,
        'missing_critical_files': critical,
        'missing_modules': [{'module': name, 'imported_by': sorted(origins)}
                            for name, origins in sorted(missing.items())],
        'syntax_errors': syntax_errors, 'warnings': warnings,
        'note': 'Original snapshots were not modified. Existing source was not overwritten. No database or model access. Static audit does not prove full runtime compatibility.',
    }
    report_path = backend / 'vscode_recovery_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Restored missing files:', len(restored))
    print('Existing files preserved:', len(preserved))
    print('Missing critical files:', ', '.join(critical) or 'none')
    print('Missing imported modules:', len(missing))
    print('Report:', report_path)
    print('Backup:', backup)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, type=Path)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--before', help='Optional local saved-at cutoff, e.g. 2026-10-29T09:00:00')
    parser.add_argument('--resource-root', help='Original Windows app path if the project was moved')
    args = parser.parse_args()
    if args.history is None:
        appdata = os.environ.get('APPDATA')
        if not appdata:
            parser.error('APPDATA is not set; provide --history explicitly')
        args.history = Path(appdata) / 'Code' / 'User' / 'History'
    resource_root = PureWindowsPath(args.resource_root or str(args.backend.resolve() / 'app'))
    cutoff = datetime.fromisoformat(args.before).timestamp() * 1000 if args.before else None
    restore(args.backend, args.history, resource_root, cutoff)


if __name__ == '__main__':
    main()
