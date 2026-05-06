#!/usr/bin/env bash
# Build script for Mnemosyne binary package.
# Creates a standalone executable using PyInstaller.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DIST_DIR="$SCRIPT_DIR/dist"
PACKAGE_DIR="$SCRIPT_DIR/mnemosyne"

echo "=== Mnemosyne Build Script ==="
echo ""

# Clean previous builds
rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

# Create PyInstaller spec file
cat > "$BUILD_DIR/mnemosyne.spec" << 'SPEC_EOF'
# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

# PyInstaller needs to know about sqlite-vec extension
block_cipher = None

# Get package dir
package_dir = os.path.join(os.getcwd(), 'mnemosyne')

a = Analysis(
    [os.path.join(package_dir, 'core', 'mcp_server.py')],
    pathex=[package_dir],
    binaries=[
        # Include sqlite-vec extension
        ('/home/filip/.hermes/hermes-agent/venv/lib/python3.11/site-packages/sqlite_vec/vec0', 'sqlite_vec'),
    ],
    datas=[
        # Include embedding models cache dir marker
        ('mnemosyne/models', 'mnemosyne/models'),
    ],
    hiddenimports=[
        'kuzu',
        'fastembed',
        'sqlite_vec',
        'numpy',
        'mcp',
        'mcp.server',
        'mcp.types',
        'mcp.server.stdio',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='mnemosyne',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
SPEC_EOF

echo "Created PyInstaller spec file"
echo ""
echo "To build:"
echo "  pip install pyinstaller"
echo "  pyinstaller build/mnemosyne.spec"
echo "  → Binary at: dist/mnemosyne"
echo ""
echo "Note: For open source distribution, prefer:"
echo "  pip install -e ."
echo "  mnemosyne stats"
