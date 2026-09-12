import io
import os
import zipfile
from jinja2 import Environment, FileSystemLoader
from typing import List, Set, Tuple, Dict, Any
from app.models.config import PipelineConfig

ALLOWED_COMPONENTS = {
    "parser",
    "chunking",
    "embeddings",
    "retrieval",
    "reranking",
    "generation",
    "evaluation",
    "fastapi",
    "web_ui"
}

COMPONENT_DEPENDENCIES = {
    "reranking": ["retrieval", "embeddings"],
    "retrieval": ["embeddings"],
    "generation": ["retrieval", "embeddings"],
    "evaluation": ["generation", "retrieval", "embeddings"],
    "chunking": ["parser"],
    "web_ui": ["fastapi"]
}

COMPONENT_REQUIREMENTS = {
    "parser": ["pypdf>=4.1.0"],
    "chunking": [],
    "embeddings": ["sentence-transformers>=2.5.0", "numpy>=1.26.0"],
    "retrieval": ["faiss-cpu>=1.8.0", "numpy>=1.26.0", "sentence-transformers>=2.5.0"],
    "reranking": ["sentence-transformers>=2.5.0"],
    "generation": ["httpx>=0.27.0"],
    "evaluation": ["numpy>=1.26.0", "scikit-learn>=1.4.0"],
    "fastapi": ["fastapi>=0.110.0", "uvicorn[standard]>=0.28.0", "python-multipart>=0.0.9"],
    "web_ui": ["jinja2>=3.1.3"]
}


class CodeGeneratorService:
    @staticmethod
    def resolve_dependencies(selected_components: List[str]) -> Tuple[Set[str], List[str]]:
        """
        Validate against ALLOWED_COMPONENTS allowlist and automatically resolve dependencies.
        Returns:
            Tuple of (resolved_components_set, list_of_dependency_notifications)
        """
        # 1. Filter against allowlist
        valid_selected = {c for c in selected_components if c in ALLOWED_COMPONENTS}
        resolved = set(valid_selected)
        notifications = []

        # 2. Iteratively resolve dependencies
        changed = True
        while changed:
            changed = False
            for comp in list(resolved):
                for dep in COMPONENT_DEPENDENCIES.get(comp, []):
                    if dep not in resolved:
                        resolved.add(dep)
                        notifications.append(
                            f"{comp.capitalize()} requires {dep.capitalize()}. Automatically included {dep.capitalize()}."
                        )
                        changed = True

        return resolved, notifications

    @staticmethod
    def generate_requirements(components: Set[str]) -> str:
        reqs = set()
        for comp in components:
            reqs.update(COMPONENT_REQUIREMENTS.get(comp, []))
        return "\n".join(sorted(reqs)) + "\n"

    @classmethod
    def generate_single_file(cls, components: Set[str], config: PipelineConfig) -> str:
        """Render single Python file using Jinja2 template."""
        templates_dir = os.path.join(os.path.dirname(__file__), "..", "code_templates", "single_file")
        env = Environment(loader=FileSystemLoader(templates_dir), autoescape=False)
        template = env.get_template("template.py.jinja")

        return template.render(
            components=components,
            config=config
        )

    @classmethod
    def generate_project_zip(cls, components: Set[str], config: PipelineConfig) -> io.BytesIO:
        """
        Generate complete project ZIP with clean modular structure.
        """
        single_code = cls.generate_single_file(components, config)
        requirements_txt = cls.generate_requirements(components)

        env_example = (
            "# Environment configuration for Self-Correcting RAG\n"
            "GROQ_API_KEY=your_groq_api_key_here\n"
            "DATABASE_URL=postgresql://postgres@localhost:5432/rag_db\n"
        )

        readme_md = (
            f"# Self-Correcting RAG - Exported Pipeline\n\n"
            f"Configured with:\n"
            f"- Chunk Size: {config.chunk_size}\n"
            f"- Chunk Overlap: {config.chunk_overlap}\n"
            f"- Embedding Model: {config.embedding_model}\n"
            f"- Top-K: {config.top_k}\n"
            f"- Reranking Enabled: {config.reranking_enabled}\n\n"
            f"## Quickstart\n\n"
            f"1. Create virtual environment:\n"
            f"```bash\n"
            f"python -m venv venv\n"
            f"source venv/bin/activate  # On Windows: venv\\Scripts\\activate\n"
            f"pip install -r requirements.txt\n"
            f"```\n\n"
            f"2. Set your Groq API key:\n"
            f"```bash\n"
            f"export GROQ_API_KEY=\"your_groq_api_key\"\n"
            f"```\n\n"
            f"3. Run pipeline:\n"
            f"```bash\n"
            f"python app/main.py sample.pdf \"What is the main topic?\"\n"
            f"```\n"
        )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("self-correcting-rag/app/main.py", single_code)
            zf.writestr("self-correcting-rag/requirements.txt", requirements_txt)
            zf.writestr("self-correcting-rag/.env.example", env_example)
            zf.writestr("self-correcting-rag/README.md", readme_md)

        zip_buffer.seek(0)
        return zip_buffer
