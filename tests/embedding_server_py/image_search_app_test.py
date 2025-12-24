"""Image Search Application with Embedding-based Ranking.

This application provides a GUI for searching images using text queries.
It uses the embedding server to convert both images and text into vectors,
then ranks images by similarity to the query.

Usage:
    uv run python tests/embedding_server_py/image_search_app_test.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any
import urllib.request
import urllib.error

import numpy as np
from PIL import Image, ImageTk


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# Server configuration
DEFAULT_SERVER_URL = "http://127.0.0.1:3752"
MODEL_NAME = "siglip2"


def _urljoin(base: str, path: str) -> str:
    """Join URL base and path."""
    return base.rstrip("/") + path


def _http_post_json(url: str, data: dict[str, Any], timeout: float = 120) -> tuple[int, dict | None, str]:
    """Post JSON data and return (status, parsed_response, raw_body)."""
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = None
            return getattr(resp, "status", 200), parsed, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else str(e)
        return e.code, None, body


def _make_file_uri(file_path: str) -> str:
    """Convert file path to file:// URI."""
    return Path(file_path).resolve().as_uri()


def _image_to_base64_data_uri(image_path: str, max_size: int = 512) -> str:
    """Convert image file to base64 data URI, with optional resizing."""
    try:
        img = Image.open(image_path)
        # Resize if too large to save bandwidth
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buf = io.BytesIO()
        # Convert to RGB if needed (e.g., for RGBA images)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"Error converting {image_path} to base64: {e}")
        raise


def start_embedding_server(server_url: str) -> subprocess.Popen:
    """Start the embedding server as a subprocess and wait for it to be ready."""
    server_script = Path("src/embedding_server_py/main_es.py")
    if not server_script.exists():
        # Try relative to this script
        server_script = Path(__file__).parent.parent.parent / "src" / "embedding_server_py" / "main_es.py"
    
    if not server_script.exists():
        raise FileNotFoundError(f"Server script not found: {server_script}")
    
    print(f"Starting embedding server: {server_script}")
    env = os.environ.copy()
    env.setdefault("EMBEDDING_SERVER_ALLOW_LOCAL_FILE_URIS", "1")
    
    process = subprocess.Popen(
        [sys.executable, str(server_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    
    # Print server output in background
    def print_output():
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                if line:
                    print(f"[SERVER] {line.rstrip()}")
    
    threading.Thread(target=print_output, daemon=True).start()
    
    # Wait for server to be ready
    health_url = _urljoin(server_url, "/health")
    max_retries = 60
    print(f"Waiting for server at {health_url}...")
    
    for i in range(max_retries):
        if process.poll() is not None:
            raise RuntimeError(f"Server exited prematurely with code {process.returncode}")
        
        try:
            with urllib.request.urlopen(health_url, timeout=1) as resp:
                if resp.status == 200:
                    print("Server is ready!")
                    return process
        except Exception:
            time.sleep(1)
            if i % 10 == 0 and i > 0:
                print(f"Still waiting... ({i}/{max_retries})")
    
    process.terminate()
    raise RuntimeError("Server failed to start in time")


class ImageSearchApp:
    """GUI application for image search using embeddings."""
    
    def __init__(self, root: tk.Tk, server_url: str):
        self.root = root
        self.server_url = server_url
        self.root.title("Image Search with Embeddings")
        self.root.geometry("1000x700")
        
        # Data storage
        self.image_paths: list[str] = []
        self.image_embeddings: list[np.ndarray] = []
        self.ranked_indices: list[int] = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Top frame: Directory selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Image Directory:").pack(side=tk.LEFT)
        self.dir_label = ttk.Label(top_frame, text="No directory selected", foreground="gray")
        self.dir_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(top_frame, text="Select Directory", command=self._select_directory).pack(side=tk.LEFT)
        ttk.Button(top_frame, text="Process Images", command=self._process_images).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(top_frame, text="", foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Middle frame: Search box
        search_frame = ttk.Frame(self.root, padding="10")
        search_frame.pack(fill=tk.X)
        
        ttk.Label(search_frame, text="Search Query:").pack(side=tk.LEFT)
        self.query_entry = ttk.Entry(search_frame, width=50)
        self.query_entry.pack(side=tk.LEFT, padx=10)
        self.query_entry.bind("<Return>", lambda e: self._search())
        
        ttk.Button(search_frame, text="Search", command=self._search).pack(side=tk.LEFT)
        
        # Bottom frame: Results display with scrollbar
        results_frame = ttk.Frame(self.root, padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(results_frame)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.results_container = ttk.Frame(canvas)
        
        self.results_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.results_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = canvas
    
    def _select_directory(self):
        """Open directory selection dialog."""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.dir_label.config(text=directory, foreground="black")
            self.selected_dir = directory
            # Clear previous data
            self.image_paths = []
            self.image_embeddings = []
            self.ranked_indices = []
            self._clear_results()
    
    def _scan_images(self, directory: str) -> list[str]:
        """Scan directory for image files."""
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(root, file))
        return sorted(image_files)
    
    def _process_images(self):
        """Process all images in the selected directory."""
        if not hasattr(self, "selected_dir"):
            messagebox.showwarning("No Directory", "Please select a directory first.")
            return
        
        self.status_label.config(text="Scanning images...")
        self.root.update()
        
        # Scan for images
        self.image_paths = self._scan_images(self.selected_dir)
        
        if not self.image_paths:
            messagebox.showinfo("No Images", "No images found in the selected directory.")
            self.status_label.config(text="")
            return
        
        self.status_label.config(text=f"Found {len(self.image_paths)} images. Processing...")
        self.root.update()
        
        # Process in background thread
        threading.Thread(target=self._process_images_thread, daemon=True).start()
    
    def _process_images_thread(self):
        """Process images in a background thread."""
        try:
            # Batch size for processing
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(self.image_paths), batch_size):
                batch = self.image_paths[i:i+batch_size]
                self.root.after(0, lambda i=i: self.status_label.config(
                    text=f"Processing images {i+1}-{min(i+batch_size, len(self.image_paths))} of {len(self.image_paths)}..."
                ))
                
                # Convert to file URIs
                inputs = [_make_file_uri(path) for path in batch]
                print(inputs)
                
                # Get embeddings
                url = _urljoin(self.server_url, "/v1/embeddings")
                status, parsed, raw = _http_post_json(
                    url,
                    {"input": inputs, "model": MODEL_NAME, "encoding_format": "float"},
                    timeout=1200
                )
                
                if status != 200 or not parsed:
                    raise RuntimeError(f"Failed to get embeddings: {raw}")
                
                # Extract embeddings
                for item in parsed["data"]:
                    all_embeddings.append(np.array(item["embedding"], dtype=np.float32))
            
            self.image_embeddings = all_embeddings
            self.root.after(0, lambda: self.status_label.config(
                text=f"Processed {len(self.image_paths)} images successfully!",
                foreground="green"
            ))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to process images: {e}"))
            self.root.after(0, lambda: self.status_label.config(text="Error processing images", foreground="red"))
    
    def _search(self):
        """Search images based on text query."""
        if not self.image_embeddings:
            messagebox.showwarning("No Images", "Please process images first.")
            return
        
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("No Query", "Please enter a search query.")
            return
        
        self.status_label.config(text="Searching...", foreground="blue")
        self.root.update()
        
        # Get query embedding
        try:
            url = _urljoin(self.server_url, "/v1/embeddings")
            status, parsed, raw = _http_post_json(
                url,
                {"input": [query], "model": MODEL_NAME, "encoding_format": "float"},
                timeout=30
            )
            
            if status != 200 or not parsed:
                raise RuntimeError(f"Failed to get query embedding: {raw}")
            
            query_embedding = np.array(parsed["data"][0]["embedding"], dtype=np.float32)
            
            # Calculate similarities (dot product since embeddings are normalized)
            similarities = [
                np.dot(query_embedding, img_emb)
                for img_emb in self.image_embeddings
            ]
            
            # Rank by similarity (descending)
            self.ranked_indices = sorted(
                range(len(similarities)),
                key=lambda i: similarities[i],
                reverse=True
            )
            
            # Display results
            self._display_results(similarities)
            self.status_label.config(text=f"Found {len(self.ranked_indices)} results", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Search Error", f"Failed to search: {e}")
            self.status_label.config(text="Search failed", foreground="red")
    
    def _display_results(self, similarities: list[float]):
        """Display ranked image results."""
        self._clear_results()
        
        # Show top 20 results
        for rank, idx in enumerate(self.ranked_indices[:50], 1):
            self._add_result_item(rank, idx, similarities[idx])
        
        # Reset scroll to top
        self.canvas.yview_moveto(0)
    
    def _add_result_item(self, rank: int, image_idx: int, similarity: float):
        """Add a single result item to the results container."""
        frame = ttk.Frame(self.results_container, padding="5", relief=tk.RIDGE, borderwidth=1)
        frame.pack(fill=tk.X, pady=2)
        
        # Rank and similarity
        info_text = f"#{rank}  Similarity: {similarity:.4f}"
        ttk.Label(frame, text=info_text, font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        # Image path
        path = self.image_paths[image_idx]
        ttk.Label(frame, text=path, font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        
        # Thumbnail
        try:
            img = Image.open(path)
            img.thumbnail((150, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(frame, image=photo)
            label.image = photo  # Keep a reference
            label.pack(anchor=tk.W, pady=5)
        except Exception as e:
            ttk.Label(frame, text=f"[Could not load image: {e}]", foreground="red").pack(anchor=tk.W)
    
    def _clear_results(self):
        """Clear all result items."""
        for widget in self.results_container.winfo_children():
            widget.destroy()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Image Search Application")
    parser.add_argument("--server_url", default=DEFAULT_SERVER_URL, help="Embedding server URL")
    parser.add_argument("--no_start_server", action="store_true", help="Don't start server (assume already running)")
    args = parser.parse_args()
    
    server_proc = None
    
    try:
        # Start server if needed
        if not args.no_start_server:
            print("Starting embedding server...")
            server_proc = start_embedding_server(args.server_url)
            print("Server started successfully!")
        
        # Create and run GUI
        root = tk.Tk()
        app = ImageSearchApp(root, args.server_url)
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        if server_proc:
            print("Stopping server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
            print("Server stopped.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())